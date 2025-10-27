<p align="right">
  <a href="README.zh-CN.md"><img alt="中文" src="https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-red"></a>
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/lang-English-blue"></a>
</p>

# AEM Neutron → PINN

Data-first repo for **2D neutron imaging** of AEM electrolyzers and weakly supervised **PINN** modeling.

## What you get
- **Alignment**: deskew + window rectification → pixel-to-pixel comparable frames.
- **Two-level ROIs**: *Window ROI* (for **OD/α**) and *Flowfield ROI* (for channel stats).
- **OD → α**: `OD = -ln(I/I_ref)` → relative/absolute α (with `mu_w`, `H`).
- **Batch metrics**: mean α (̄α), heterogeneity (η), axial decay length (Ld), bridging flag (B).

## Quick start
```bash
pip install -r requirements.txt
python scripts/batch_process.py \ 
  --in_dir data/geometry_A \ 
  --ref_glob '*ref*.png' \ 
  --img_glob '*.png' \ 
  --out_dir outputs/geometry_A
```

**Outputs**
- `*_warped_window.png`  – rectified window (use for OD/α)
- `*_alpha_rel.png`      – relative α map (0–1)
- `metrics.csv`          – ̄α, η, Ld, B per image

**Notes**
- Use **one reference frame per geometry**.
- Compute OD/α inside *Window ROI*; compute statistics inside *Flowfield ROI*.
- Provide `mu_w` and `H` to export absolute α; otherwise a relative α proxy is exported.

## Roadmap: PINN module
PDE residuals + boundary + **pixel α loss** + **Faraday consistency**; parameters: `k_eff`, rel. permeability exponent, detachment/slip, `kappa_eff`, `D_eff`.
