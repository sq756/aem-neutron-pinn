# aem-neutron-pinn

Data-first repo for 2D neutron imaging of AEM electrolyzers.

**Goals**
- Align frames (deskew + window rectification) so pixels are comparable across operating points.
- Two-level ROIs: Window ROI (for OD/alpha) and Flowfield ROI (for channel-level stats).
- Convert OD = -ln(I/I_ref) to relative/absolute alpha.
- Batch export alpha maps and metrics (mean alpha, heterogeneity eta, axial decay length Ld, bridging flag).

**Quick start**
1) `pip install -r requirements.txt`
2) Put images of one geometry in a folder and include exactly one reference frame (filename contains `ref`).
3) Run: `python scripts/batch_process.py --in_dir your_data/geometry_A --ref_glob *ref*.png --img_glob *.png --out_dir outputs/geometry_A`

**Outputs**
- *_warped_window.png : rectified window (use for OD)
- *_alpha_rel.png     : relative alpha (0-1)
- metrics.csv         : bar_alpha, eta, Ld, bridge per image

**Notes**
- Use the same reference frame for all operating points of a given geometry.
- Compute OD/alpha inside Window ROI; compute stats inside Flowfield ROI.
- If you know mu_w (water attenuation) and H (effective thickness), the script can output absolute alpha; otherwise it returns relative alpha.

**Roadmap**
A PINN module (weakly supervised with PDE residuals + Faraday consistency + pixel alpha) will be added for parameter identification (k_eff, relative permeability exponent, detachment/slip constants, kappa_eff, D_eff).