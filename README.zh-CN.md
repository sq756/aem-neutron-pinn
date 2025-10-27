<p align="right">
  <a href="README.zh-CN.md"><img alt="中文" src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E4%B8%AD%E6%96%87-red"></a>
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/lang-English-blue"></a>
</p>

# aem-neutron-pinn（中文）

用于 **AEM 电解槽二维中子成像** 的处理与 **PINN** 建模。

## PINN（四损失 + 低维流函数）

```bash
python pinn/train_pinn.py \ 
  --alpha_img 数据/alpha.png \ 
  --roi_mask 数据/roi.png \ 
  --chi_act  数据/chi.png \ 
  --j 0.2 --Q 100 --inlet_side left
```

### 输入解释
- **alpha_img**：ROI 内逐像素的相对 α（由 OD 归一化得到，范围 [0,1]）。
- **j, Q**：本帧对应的电流密度与体积流量，都是**标量参数**。
- **chi_act**：活性区二值掩膜；若省略，默认取 ROI 全为 1。

### 每项损失在训练中的作用示意
```mermaid
flowchart LR
  A[像素: alpha_img] -- L_data --> G[alpha_theta(x,y)]
  P[PDE 物理残差] -- L_PDE --> G
  F[法拉第积分一致性
  (∑Sg 对 C(j))] -- L_F --> Params[beta, D_eff, k_out, psi]
  Q[体积流量约束
  (入口通量 对 Q)] -- L_Q --> U[psi → u]
  G --> P
  U --> P
  Params --> P
```
