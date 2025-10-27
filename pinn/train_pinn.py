import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

from .model import AlphaMLP, StreamFunctionFourier

# ------------------------------
# I/O helpers
# ------------------------------

def load_gray01(path):
    arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return arr


def to_tensor_xy(xy):
    return torch.tensor(xy, dtype=torch.float32)


def sample_points(mask, n):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError('ROI/mask is empty.')
    idx = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
    X = xs[idx]; Y = ys[idx]
    H, W = mask.shape
    xy = np.stack([X/W, Y/H], axis=1)
    return to_tensor_xy(xy)


def boundary_points(mask, side='left', n=1024):
    H,W = mask.shape
    if side=='left':
        xs = np.zeros(n); ys = np.linspace(0,H-1,n)
    elif side=='right':
        xs = np.ones(n)*(W-1); ys = np.linspace(0,H-1,n)
    elif side=='top':
        xs = np.linspace(0,W-1,n); ys = np.zeros(n)
    else: # bottom
        xs = np.linspace(0,W-1,n); ys = np.ones(n)*(H-1)
    xy = np.stack([xs/W, ys/H], axis=1)
    return to_tensor_xy(xy)


def alpha_lookup(alpha_img01, xy):
    H, W = alpha_img01.shape
    x = (xy[:,0].detach().cpu().numpy() * W).clip(0, W-1)
    y = (xy[:,1].detach().cpu().numpy() * H).clip(0, H-1)
    vals = alpha_img01[y.astype(int), x.astype(int)]
    return torch.tensor(vals, dtype=torch.float32).unsqueeze(1)

# ------------------------------
# Training core
# ------------------------------

def train_one(
    alpha_img_path, roi_mask_path, chi_act_path=None,
    j=0.2, Q=100.0, inlet_side='left',
    device='cpu', epochs_warm=300, epochs_full=1000,
    n_data=20000, n_pde=20000, seed=0,
    save_dir=None, c_j_scale=1.0
):
    np.random.seed(seed); torch.manual_seed(seed)

    alpha_img = load_gray01(alpha_img_path)
    roi = (load_gray01(roi_mask_path) > 0.5).astype(np.uint8)
    chi_act = (load_gray01(chi_act_path) > 0.5).astype(np.uint8) if chi_act_path else roi
    H,W = roi.shape

    alpha_net = AlphaMLP().to(device)
    psi_net = StreamFunctionFourier().to(device)
    log_D = nn.Parameter(torch.tensor(-2.3))  # ~0.1
    log_k = nn.Parameter(torch.tensor(-2.3))  # ~0.1
    log_b = nn.Parameter(torch.tensor(0.0))   # ~1.0
    params = list(alpha_net.parameters()) + list(psi_net.parameters()) + [log_D, log_k, log_b]
    opt = optim.Adam(params, lr=1e-3)

    chi_tensor = torch.tensor(chi_act, dtype=torch.float32, device=device)

    def losses(w_data, w_pde, w_F, w_Q):
        # sample points
        xy_d = sample_points(roi, n_data).to(device).requires_grad_(True)
        xy_p = sample_points(roi, n_pde).to(device).requires_grad_(True)
        xy_in = boundary_points(roi, inlet_side, n=2048).to(device).requires_grad_(True)

        # Data loss
        a_pred = alpha_net(xy_d)
        a_true = alpha_lookup(alpha_img, xy_d).to(device)
        L_data = torch.mean((a_pred - a_true)**2)

        # PDE residuals
        a_p = alpha_net(xy_p)
        _, u_p = psi_net(xy_p)
        D = torch.nn.functional.softplus(log_D)
        k = torch.nn.functional.softplus(log_k)
        beta = torch.nn.functional.softplus(log_b)

        grads = torch.autograd.grad(a_p, xy_p, grad_outputs=torch.ones_like(a_p), create_graph=True)[0]
        ax = grads[:,0:1]; ay = grads[:,1:2]
        axx = torch.autograd.grad(ax, xy_p, torch.ones_like(ax), create_graph=True)[0][:,0:1]
        ayy = torch.autograd.grad(ay, xy_p, torch.ones_like(ay), create_graph=True)[0][:,1:2]
        lap = axx + ayy
        conv = u_p[:,0:1]*ax + u_p[:,1:2]*ay

        X = (xy_p[:,0]*W).long().clamp(0,W-1)
        Y = (xy_p[:,1]*H).long().clamp(0,H-1)
        chi_vals = chi_tensor[Y, X].unsqueeze(1)
        Sg = beta * j * chi_vals
        res = conv - D*lap + k*a_p - Sg
        L_pde = torch.mean(res**2)

        # Faraday consistency (Monte Carlo over ROI)
        Cj = c_j_scale * j
        L_F = (torch.mean(Sg) - Cj)**2

        # Flow-rate consistency (inlet normal flux)
        _, u_in = psi_net(xy_in)
        if inlet_side in ['left','right']:
            nvec = torch.tensor([[1.0,0.0]] if inlet_side=='left' else [[-1.0,0.0]], device=device)
        else:
            nvec = torch.tensor([[0.0,1.0]] if inlet_side=='bottom' else [[0.0,-1.0]], device=device)
        un = torch.sum(u_in * nvec, dim=1, keepdim=True)
        U_target = Q / (Q + 1e-8)  # nondim anchor in (0,1)
        L_Q = (torch.mean(un) - U_target)**2

        L = w_data*L_data + w_pde*L_pde + w_F*L_F + w_Q*L_Q
        scalars = dict(L_data=L_data.item(), L_pde=L_pde.item(), L_F=L_F.item(), L_Q=L_Q.item(), D=D.item(), k=k.item(), beta=beta.item())
        return L, scalars

    hist = []
    # warm-up
    for ep in range(epochs_warm):
        opt.zero_grad()
        L, s = losses(1.0, 0.0, 0.0, 0.0)
        L.backward(); opt.step()
        if (ep+1) % 50 == 0: hist.append((ep+1, s))

    # full
    for ep in range(epochs_full):
        opt.zero_grad()
        w_pde = min(1.0, 0.1 + ep/epochs_full)
        L, s = losses(1.0, w_pde, 0.5, 0.5)
        L.backward(); opt.step()
        if (ep+1) % 50 == 0: hist.append((epochs_warm+ep+1, s))

    # export predictions on a dense grid
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        xs = np.linspace(0,1,W); ys = np.linspace(0,1,H)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1,2)
        xy = torch.tensor(grid, dtype=torch.float32, device=device)
        with torch.no_grad():
            a_grid = alpha_net(xy).reshape(H,W).cpu().numpy()
            _, u_grid = psi_net(xy); u_grid = u_grid.reshape(H,W,2).cpu().numpy()
        np.save(Path(save_dir)/'alpha_pred.npy', a_grid)
        np.save(Path(save_dir)/'u_pred.npy', u_grid)
        Image.fromarray((np.clip(a_grid,0,1)*255).astype(np.uint8)).save(Path(save_dir)/'alpha_pred.png')

    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha_img', required=True)
    ap.add_argument('--roi_mask', required=True)
    ap.add_argument('--chi_act', default=None)
    ap.add_argument('--j', type=float, required=True)
    ap.add_argument('--Q', type=float, required=True)
    ap.add_argument('--inlet_side', choices=['left','right','top','bottom'], default='left')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--save_dir', default='pinn_outputs')

    args = ap.parse_args()

    train_one(
        args.alpha_img, args.roi_mask, args.chi_act, j=args.j, Q=args.Q, inlet_side=args.inlet_side, device=args.device, save_dir=args.save_dir
    )
