<p align="right">
  <a href="README.zh-CN.md"><img alt="中文" src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E4%B8%AD%E6%96%87-red"></a>
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/lang-English-blue"></a>
</p>

# aem-neutron-pinn（中文）

这个仓库用于 **AEM 电解槽的二维中子成像** 处理与 **PINN（物理约束神经网络）** 的弱监督建模。

## 你可以得到
- **统一对齐**：自动去倾斜 + 检测内腔窗口并透视矫正，保证不同工况像素可比；
- **双层 ROI**：`Window ROI`（用于 **OD/α**）与 `Flowfield ROI`（用于通道级统计）；
- **OD→α**：`OD=-ln(I/I_ref)` → 相对/绝对 α（需 `mu_w` 与 `H` 才能得到绝对 α）；
- **批处理指标**：̄α、η、轴向衰减长度 Ld、桥接标记 B。

## 快速开始
```bash
pip install -r requirements.txt
python scripts/batch_process.py \ 
  --in_dir data/geometry_A \ 
  --ref_glob '*ref*.png' \ 
  --img_glob '*.png' \ 
  --out_dir outputs/geometry_A
```

**输出**
- `*_warped_window.png`：窗口拉正图（用于 OD/α）；
- `*_alpha_rel.png`：相对 α 图（0–1）；
- `metrics.csv`：每张图的 ̄α、η、Ld、B。

**注意**
- 每个几何只用 **1 张参考帧**；
- *Window ROI* 内做 OD/α，*Flowfield ROI* 内做统计；
- 若提供 `mu_w`（水的衰减系数）与 `H`（等效厚度）可导出**绝对 α**，否则为**相对 α**。

## 规划：PINN 模块
损失 = PDE 残差 + 边界 + **像素级 α 数据项** + **法拉第积分一致性**；可识别参数：`k_eff`、相对渗透率指数、脱附/滑移常数、`kappa_eff`、`D_eff` 等。
