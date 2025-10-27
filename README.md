# aem-neutron-pinn

<p align="right">
  <a href="README.zh-CN.md"><img alt="中文" src="https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-red"></a>
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/lang-English-blue"></a>
</p>

Data-first repo for 2D neutron imaging of AEM electrolyzers and weakly supervised PINN modeling.

## PINN (four losses + low-dim stream function)

```bash
python pinn/train_pinn.py \ 
  --alpha_img path/to/alpha.png \ 
  --roi_mask path/to/roi.png \ 
  --chi_act path/to/chi.png \ 
  --j 0.2 --Q 100 --inlet_side left
```

### What are the inputs?
- **alpha_img**: pixelwise relative alpha in [0,1] inside the ROI (obtained from OD map).
- **j, Q**: scalars for the current density and bulk flow rate of *this frame*.
- **chi_act**: binary mask of active area; if omitted, the script uses ROI (1s everywhere).

### Loss roles
```mermaid
flowchart LR
  A[Pixels: alpha_img] -- L_data --> G[alpha_theta(x,y)]
  P[PDE residual] -- L_PDE --> G
  F[Faraday integral
  (sum Sg vs C(j))] -- L_F --> Params[beta, D_eff, k_out, psi]
  Q[Flow-rate consistency
  (inlet flux vs Q)] -- L_Q --> U[psi -> u]
  G --> P
  U --> P
  Params --> P
```
