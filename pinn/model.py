import math
import torch
import torch.nn as nn

class StreamFunctionFourier(nn.Module):
    """
    Low-dimensional stream function: psi(x,y) = sum a_{mn} sin(m*pi*x) sin(n*pi*y).
    Returns psi and divergence-free velocity u = (dpsi/dy, -dpsi/dx).
    ""
    def __init__(self, modes=((1,1),(2,1),(1,2),(2,2),(3,1),(1,3))):
        super().__init__()
        self.register_buffer('modes', torch.tensor(modes, dtype=torch.float32))
        self.coeff = nn.Parameter(torch.zeros(len(modes)))

    def forward(self, xy):
        x = xy[:,0:1]; y = xy[:,1:2]
        pi = math.pi
        # psi
        psi = torch.zeros_like(x)
        for i,(m,n) in enumerate(self.modes):
            psi = psi + self.coeff[i] * torch.sin(pi*m*x) * torch.sin(pi*n*y)
        # gradients
        grads = torch.autograd.grad(psi, xy, grad_outputs=torch.ones_like(psi),
                                    create_graph=True, retain_graph=True)[0]
        dpsi_dx = grads[:,0:1]; dpsi_dy = grads[:,1:2]
        u = torch.cat([dpsi_dy, -dpsi_dx], dim=1)
        return psi, u

class AlphaMLP(nn.Module):
    def __init__(self, width=128, depth=4, out_act='sigmoid'):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.Tanh()]
            in_dim = width
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
        self.out_act = out_act
    def forward(self, xy):
        a = self.net(xy)
        if self.out_act == 'sigmoid':
            a = torch.sigmoid(a)
        return a
