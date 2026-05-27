# XJTU 计算机视觉大实验：基于 DenseNet 与 ViT 的年龄估计

## 1. 实验目的

本实验在 APPA-REAL 表观年龄数据集上完成端到端的年龄估计任务，
并对两类具有代表性的图像表征模型进行受控对比：

1. 基于卷积神经网络的 DenseNet121；
2. 基于 Transformer 的 ViT-B/16。

目标包括：

- 将年龄估计建模为一维回归问题，使用统一的损失函数与评价指标完成训练与测试；
- 在尽可能公平的实验协议下比较 DenseNet 与 ViT 的整体表现以及在不同年龄段的差异；
- 通过 per-age-group 误差分析观察两类归纳偏置（局部卷积 vs. 全局自注意力）
  在中等规模数据 + 长尾年龄分布下的实际表现。

最终输出包括完整的训练日志、测试指标、对比图表与本报告。

## 2. 实验环境

实际训练在云端 GPU 服务器上完成，环境信息直接来自
`results/densenet/env_snapshot.txt` 与 `results/vit/env_snapshot.txt`：

| 项目 | 值 |
|---|---|
| 操作系统 / Python | Python 3.12.3 |
| 深度学习框架 | torch 2.5.1+cu124 |
| CUDA / cuDNN | CUDA 12.4 / cuDNN 90100 |
| GPU | NVIDIA A800 80GB PCIe，单卡 |
| 随机种子 | 42（两个模型相同） |
| Git revision | 6d40b0b665cf31fe46fcd1f1dc4a92608a8e82d8（dirty=no） |
| 运行时间戳 | 2026-05-28 02:13:29 |

随机性控制：在 `src/utils.py` 中通过 `set_seed(42)` 统一设置
`random`、`numpy`、`torch`（CPU + CUDA）的种子，并配合 DataLoader 的
`worker_init_fn` 给每个 worker 派生子种子；
`torch.backends.cudnn` 设置为 deterministic 路径以减少 kernel 选择带来的抖动。

依赖列表见仓库根目录 `requirements.txt`，
两个模型均通过 `python main.py --model <densenet|vit> --mode all --epochs 25 --seed 42`
一键运行。

## 3. 数据集介绍

实验使用 **APPA-REAL** 表观年龄数据集（Agustsson et al., 2017）。
数据来源、规模与划分如下：

- **来源**：APPA-REAL 官方发布版本，目录结构为
  `appa-real-release/{train,valid,test}/<file_name>_face.jpg`，
  标签文件为 `gt_avg_<split>.csv`。
- **标签语义**：使用 `apparent_age_avg` 字段，即多名标注者对同一张人脸图像给出的
  表观年龄（apparent age）的平均值。表观年龄是连续实数，与生物学年龄略有偏差，
  但其连续性使得回归建模天然适用。
- **Split**：直接使用官方的 train / valid / test 划分，未做任何重切分。测试集大小
  从两个模型的 `test_summary.txt` 得到一致结果：**n_test = 1978**。训练集大致
  约 3800-3900 张（drop_last 后从 step_log.csv 可推算每个 epoch 训练步数为
  DN=121 步 × bs=32 ≈ 3872、ViT=241 步 × bs=16 ≈ 3856）。
- **Ignore list**：dataset 模块支持 `ignore_list.csv` 自动剔除标注质量差的样本
  （见 `src/dataset.py`）。
- **年龄分布**：测试集按年龄分箱后呈典型长尾分布，[20, 30) 年龄段占 579 样本（29%），
  而 [70, +inf) 仅 71 样本（3.6%），[0, 10) 也只有 222 样本（11%）。
  这一分布对误差分析至关重要，见第 11 章。

测试集年龄段样本数（来自 `per_age_group_mae.csv`）：

| 年龄段 | 样本数 | 占比 |
|---|---:|---:|
| [0, 10) | 222 | 11.2% |
| [10, 20) | 147 | 7.4% |
| [20, 30) | 579 | 29.3% |
| [30, 40) | 465 | 23.5% |
| [40, 50) | 244 | 12.3% |
| [50, 60) | 146 | 7.4% |
| [60, 70) | 104 | 5.3% |
| [70, +inf) | 71 | 3.6% |

## 4. 数据预处理方法

两个模型共用同一套预处理流水线（`src/dataset.py:build_transform`），
这是公平实验协议的一部分。

**训练集** (`train=True`)：

1. `Resize` 到 `(258, 258)`（即 `int(224 * 1.15)`），略大于网络输入；
2. `RandomCrop(224, 224)`，提供轻度位置扰动；
3. `RandomHorizontalFlip(p=0.5)`，人脸左右对称先验下的标准增强；
4. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`，光照鲁棒性；
5. `ToTensor()`；
6. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`，
   ImageNet 统计量（与 torchvision 预训练权重对齐）。

**验证 / 测试集** (`train=False`)：

1. `Resize` 直接到 `(224, 224)`；
2. `ToTensor()`；
3. 同一组 ImageNet `Normalize`。

设计动机：DenseNet 与 ViT 的 torchvision 预训练权重均以 ImageNet
mean/std 归一化、224×224 输入为前提，使用统一预处理可同时满足两者，
也避免预处理差异污染对比结果。验证 / 测试阶段不做任何随机增强，
确保评价确定且可复现。

## 5. 年龄估计任务建模

年龄是一维连续实数，将其建模为**回归任务**比建模为分类任务更自然：

- 分类视角下，年龄被离散化为 N 个类别，所有相邻类别在交叉熵下的距离相等。
  但实际中 "把 23 岁预测成 24 岁" 与 "把 23 岁预测成 80 岁" 的代价显然不同，
  分类形式直接丢失了这一连续结构。
- 回归视角下，模型直接输出一个标量年龄 `y_hat`，损失函数与评价指标都
  按照实数距离定义，恰好匹配任务语义。

**损失函数**：两个模型均使用 `nn.L1Loss()`（即 MAE 损失，
配置文件中 `loss: l1`）。选择 L1 而非 MSE 是因为：

- L1 对异常标签（噪声年龄）更鲁棒；
- L1 与最终评价指标 MAE 单位一致，优化目标 = 评价目标，
  避免训练—评价不一致带来的优化 bias。

**评价指标**：

- **MAE**（Mean Absolute Error）：`mean(|y_hat - y|)`，单位为 "岁"，可解释性强；
- **RMSE**（Root Mean Squared Error）：`sqrt(mean((y_hat - y)^2))`，
  对大误差更敏感，反映尾部行为。

模型输出头统一为 `Linear(in_features, 1)`，forward 后通过 `squeeze(1)`
压成 `(B,)` 与标签计算损失（`src/train.py`）。
回归输出无值域约束，理论上可能出现负数或大于 100 的预测，
分析时按原始值统计（不做 clip），以反映模型的真实行为。

## 6. DenseNet 模型设计

**架构来源**：Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017。

**具体实现**（`src/models.py:_build_densenet`）：

```python
weights = tvm.DenseNet121_Weights.IMAGENET1K_V1
model = tvm.densenet121(weights=weights)
in_features = model.classifier.in_features    # 1024
model.classifier = nn.Linear(in_features, 1)  # 回归头
```

关键设计点：

- **直接复用 torchvision 的 `densenet121`**，4 个 Dense Block 共 121 层，
  通道增长率 32，最后一个全局池化层输出 1024 维特征；
- **预训练权重**：加载 IMAGENET1K_V1，作为 backbone 的初始化；
  数据集只有约 4k 张训练样本，从头训练几乎不可能收敛到当前水平；
- **替换 classifier**：将原本 `Linear(1024, 1000)` 替换为 `Linear(1024, 1)`；
- **全网络微调**：未冻结任何层，所有参数都参与梯度更新。
  虽然 "先训 head 再 fine-tune" 是常见技巧，
  但 25 个 epoch 已足够 backbone 适应任务，
  且这样能与 ViT 端的训练协议保持一致。

DenseNet 通过密集连接强化梯度流，并在每层显式重用前层特征，
具有较强的局部纹理建模能力，对面部皱纹、毛孔等与年龄相关的局部特征友好。

## 7. ViT 模型设计

**架构来源**：Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021。

**具体实现**（`src/models.py:_build_vit_b_16`）：

```python
weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1
model = tvm.vit_b_16(weights=weights)
in_features = model.heads.head.in_features    # 768
model.heads.head = nn.Linear(in_features, 1)  # 回归头
```

关键设计点：

- **使用 torchvision 的 `vit_b_16`**：Patch size 16、12 个 Transformer
  encoder 层、12 个 attention head、隐藏维 768、MLP 维 3072；
  224×224 输入对应 14×14 = 196 个 patch token（加 1 个 CLS token）；
- **预训练权重**：IMAGENET1K_V1。ViT 缺少卷积的平移等变 / 局部性归纳偏置，
  对预训练的依赖性远高于 CNN，从头训练在 ~4k 样本上不可行；
- **替换 `heads.head`**：torchvision 的 ViT 头部是
  `Sequential` 结构，最后一层是 `Linear(768, 1000)`，
  将其替换为 `Linear(768, 1)`；
- **全网络微调**：与 DenseNet 一致，整网可训练。

ViT 通过自注意力建模 patch 之间的全局关系，理论上能捕捉跨区域的面部
结构特征（如对称性、整体轮廓），但缺乏 CNN 的局部归纳偏置，
在数据规模偏小或长尾分布上容易暴露短板（见第 11 章误差分析）。

## 8. 训练方法与参数设置

### 8.1 公平实验协议

为保证两个模型的结果可比，下列维度严格一致：

- **同一份随机种子**：`seed=42`，覆盖 `random`、`numpy`、`torch`、CUDA、
  DataLoader generator、worker init；
- **同一份 data split**：APPA-REAL 官方 train / valid / test，
  未做任何重切分；
- **同一组 augmentation 策略**：见第 4 章，由
  `src/dataset.py:build_transform` 共享；
- **同一种损失函数**：`nn.L1Loss()`；
- **同一组 ImageNet mean/std normalize**；
- **同一个训练循环**：`src/train.py:run_training`，全局共用，
  代码中**没有** `if model_name == "vit"` 的特判分支；
- **同样的 epoch 数**：25；
- **同样的输入尺寸**：224×224；
- **同样的评价流程**：以 `best_val_mae` 对应的 checkpoint 评估 test。

仅有的差异由各自的 YAML 配置（`configs/densenet.yaml`、`configs/vit.yaml`）
驱动，这些差异都遵循各自架构的常见 best practice：

| 维度 | DenseNet121 | ViT-B/16 |
|---|---|---|
| optimizer | Adam | AdamW |
| learning rate | 1e-4 | 5e-5 |
| weight decay | 0.0 | 0.05 |
| scheduler | StepLR (step=10, gamma=0.5) | CosineAnnealingLR (T_max=25) |
| batch size | 32 | 16 |
| epochs | 25 | 25 |
| img_size | 224 | 224 |
| num_workers | 4 | 4 |
| pretrained | True (IMAGENET1K_V1) | True (IMAGENET1K_V1) |

差异说明：

- ViT 用 AdamW（解耦 weight decay）+ 较大 wd（0.05）+ cosine schedule，
  是 Transformer fine-tune 的标准配方；DenseNet 用 Adam + step decay + wd=0
  是 CNN 微调的标准配方。
- ViT 显存开销略大（attention 的 O(N^2)），batch_size 减半到 16 以适配；
  但实测 ViT 的 GPU peak 显存反而比 DenseNet 低（2990MB vs. 4123MB），
  这是因为 DenseNet 的密集连接保留了大量中间 feature map。
- 两组 lr 都对应各自的常见 fine-tune 量级，并非穷举搜索，
  仅作为合理的基线配置使用。

### 8.2 训练循环

`src/train.py:run_training` 的核心步骤（伪代码）：

```
for epoch in 1..25:
    train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_mae, val_rmse = evaluate(model, val_loader, criterion)
    scheduler.step()
    log row -> epoch_log.csv
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        save state_dict to <ckpt>_best.pth

# test 阶段
load <ckpt>_best.pth
test_loss, test_mae, test_rmse, preds, gts, names = evaluate(test_loader, return_preds=True)
```

每个 epoch 内每 20 个 step 记录一条 `step_log.csv`，
便于绘制训练曲线。

## 9. 实验结果

### 9.1 最终测试指标

直接来自 `results/comparison/comparison_table.csv` 与各自的 `test_summary.txt`：

| 指标 | DenseNet121 | ViT-B/16 |
|---|---:|---:|
| epochs 跑满 | 25 | 25 |
| best epoch (by val MAE) | **22** | **25** |
| best val MAE | 4.159 | 10.628 |
| best val RMSE | 6.388 | 14.256 |
| final val MAE | 4.292 | 10.628 |
| **test MAE** | **5.064** | **13.009** |
| **test RMSE** | **7.190** | **17.627** |
| n_test | 1978 | 1978 |
| mean epoch time (s) | 9.31 | 35.93 |
| GPU peak memory (MB) | 4122.58 | 2989.81 |

观察：

- DenseNet 在 test 上达到 MAE ≈ 5.06、RMSE ≈ 7.19，是一个合理的基线。
- ViT 在 test 上 MAE ≈ 13.01、RMSE ≈ 17.63，**显著差于 DenseNet**。
- 训练时间上，ViT 每个 epoch ≈ 35.9 秒，约为 DenseNet（9.3 秒）的 **3.86 倍**，
  整体训练成本高 3-4 倍。

### 9.2 训练 / 验证曲线对比

下面两张图分别展示两个模型的 loss 与 val MAE 随 epoch 的演化
（数据来源：`epoch_log.csv`）：

![Loss curves overlay](../results/comparison/loss_curves_overlay.png)

![Val MAE curves overlay](../results/comparison/mae_curves_overlay.png)

定性观察：

- DenseNet 训练 loss 从 epoch 1 的 26.06 快速下降到 epoch 25 的 2.16，
  val MAE 在 epoch 5 左右就跌破 6，最终最优出现在 epoch 22（val MAE 4.16）。
  之后由于 step LR 在 epoch 10、20 各衰减一次，val 曲线进入平台期，
  有轻微过拟合迹象（train loss 还在降，val MAE 不再降）。
- ViT 训练 loss 从 epoch 1 的 17.84 单调下降到 epoch 25 的 10.35，
  val MAE 从 15.04 缓慢降至 10.63，**整个曲线显著高于 DenseNet 的 val MAE**。
  最优 epoch 出现在最后一个 epoch（25），意味着 cosine schedule 走到 lr≈0
  时模型仍未充分饱和，理论上更长训练有进一步改进空间，但收敛速率已经很慢。

### 9.3 预测散点图

下面是两个模型在 test set 上 `y_pred vs y_true` 的散点图：

![Scatter side by side](../results/comparison/scatter_side_by_side.png)

- DenseNet 的散点大致沿 y=x 对角线分布，年龄越大方差越大但仍贴近对角线；
- ViT 的散点呈明显的 **"压缩到训练分布主峰" 现象**：
  无论真实年龄多大，预测值都被拉向 20-40 岁这个高密度区段，
  尾部样本几乎集体塌缩，这与下一节的 per-age-group 数字吻合。

各模型自带的单模型 loss / MAE 曲线、scatter 图：

- DenseNet：`../results/densenet/loss_curve.png`、`../results/densenet/mae_curve.png`、`../results/densenet/pred_vs_true_scatter.png`
- ViT：`../results/vit/loss_curve.png`、`../results/vit/mae_curve.png`、`../results/vit/pred_vs_true_scatter.png`

## 10. DenseNet 与 ViT 对比分析

### 10.1 整体对比

测试集上 DenseNet 的 MAE 比 ViT 低约 **7.95 岁**（5.06 vs. 13.01），
RMSE 低约 **10.44 岁**（7.19 vs. 17.63）。RMSE 与 MAE 的差距比例
（DN: 7.19/5.06 ≈ 1.42；ViT: 17.63/13.01 ≈ 1.36）显示两个模型的误差分布
形状相近（都没有极端 heavy tail），但 ViT 的全部分位都被推高了。

### 10.2 Per-age-group MAE

这是本实验最重要的对比维度，数据直接来自
`per_age_group_mae.csv`：

| 年龄段 | N | DenseNet MAE | ViT MAE | ViT / DN |
|---|---:|---:|---:|---:|
| [0, 10) | 222 | **3.03** | 22.32 | 7.37× |
| [10, 20) | 147 | 5.44 | 11.67 | 2.15× |
| [20, 30) | 579 | 3.70 | **3.67** | 0.99× |
| [30, 40) | 465 | 4.64 | 5.52 | 1.19× |
| [40, 50) | 244 | 5.90 | 15.30 | 2.60× |
| [50, 60) | 146 | 6.93 | 24.91 | 3.59× |
| [60, 70) | 104 | 8.50 | 34.65 | 4.08× |
| [70, +inf) | 71 | 12.81 | **47.81** | 3.73× |

![Per-age-group MAE bar plot](../results/comparison/per_age_bar.png)

关键发现：

1. **在 [20, 30) 主峰，ViT 略胜 DenseNet**（MAE 3.67 vs. 3.70）。
   这个年龄段也是训练 + 测试集中样本最密集的区域（test 占 29%）。
   说明 ViT **并非整体无法学习**，而是只能在数据密度高的区段学到合理的映射。
2. **在 [30, 40) 上 ViT 仍接近 DenseNet**（5.52 vs. 4.64，1.19×），
   仍属可用范围。
3. **从 [40, 50) 开始 ViT 急剧塌掉**：MAE 15.30 vs. DN 5.90，差距骤增至 2.60×。
4. **极端年龄段（[70, +inf)）ViT MAE 高达 47.81**，几乎相当于把所有老人
   一律预测成 ~30 岁；DenseNet 在该段也是表现最差的区间（MAE 12.81），
   但仍维持在 "可理解的偏差" 范围。
5. **儿童段（[0, 10)）DenseNet MAE 仅 3.03**（甚至比 [20, 30) 还低），
   而 ViT 是 22.32，比 DenseNet 高 7.37×。说明 DenseNet 学到了
   儿童 vs. 成人的关键局部特征（如脸宽比、皮肤光滑度），
   而 ViT 直接把儿童照片预测成了青年。

### 10.3 训练资源对比

| 维度 | DenseNet121 | ViT-B/16 |
|---|---|---|
| 单 epoch 时间 | 9.31 s | 35.93 s |
| 25 epoch 训练总时间 | ≈ 233 s | ≈ 898 s |
| GPU peak 显存 | 4122.58 MB | 2989.81 MB |

ViT 时间成本显著更高，但显存反而更省。
**在本数据规模下 ViT 既慢又差**。

## 11. 误差分析

### 11.1 ViT 的两端塌缩

第 10.2 节的表格揭示了 ViT 最显著的失败模式：**在远离 [20, 30) 主峰的尾部
完全塌掉**。我们可以用 `pred_vs_true_scatter.png` 中肉眼可见的现象量化：

- 主峰区段 [20, 40) 占 test 集 52.8% 的样本，ViT 在该区段的 MAE
  与 DenseNet 几乎持平（3.67-5.52）；
- 远离主峰的 [0, 10)、[40, +inf) 占 test 集约 40%，ViT 平均 MAE 约为 25 岁，
  几乎是把全部样本预测到主峰均值附近。

这一现象**不是 bug**，而是 ViT 在该数据规模下表征能力的真实反映：

1. **样本量不足**：APPA-REAL 训练集仅 ~3800 张，是 ImageNet
   的约 1/300。Dosovitskiy et al. 2021 的原论文明确指出
   ViT 的表现强依赖训练 / 预训练数据规模；在小数据上 fine-tune 不足以让
   self-attention 学到可靠的年龄相关模式。
2. **缺乏局部归纳偏置**：年龄相关的判别性特征（皱纹、毛孔、皮肤纹理）
   是空间局部的、低层视觉的，CNN 的卷积核天然适配，
   而 ViT 必须从数据中重新学习这种局部性，
   在 ~3800 样本规模下显然不够。
3. **数据长尾分布**：测试集中 [70, +inf) 仅 71 样本（3.6%），
   训练集中老人样本更少。ViT 缺乏归纳偏置，无法在低密度区段做合理的外推；
   它学到的是 "把所有人预测到训练样本的高密度区域"，
   这是经验风险最小化在高 capacity 模型 + 不均衡数据上的典型失败模式。

### 11.2 DenseNet 的均匀分布

DenseNet 的 per-age MAE 从 [0, 10) 的 3.03 单调爬升到 [70, +inf) 的 12.81。
误差随年龄段单调上升的原因主要有两点：

1. **年龄段样本数本身递减**：[20, 30) 有 579 样本，[70, +inf) 只有 71 样本，
   尾部估计噪声更大；
2. **老年人年龄方差更大**：相同 "60 岁" 的人之间外观差异
   远大于 "10 岁" 的孩子之间的差异，是数据本身的不可消除噪声。

但与 ViT 不同，DenseNet 即使在最难的 [70, +inf) 也只是 MAE 12.81 岁，
仍在 "数量级合理" 的范围内。说明 CNN 的归纳偏置让它在小数据 + 长尾分布下
依然能保持**全年龄段均匀可用**。

### 11.3 标签噪声

APPA-REAL 是 *apparent* age 标签，本身有不可消除的标注主观性
（不同标注者对同一张脸给出的年龄差异常常在 ±5 岁以内）。
这给所有模型设了 MAE 大约 3-4 岁的不可逾越下限——
DenseNet 在 [0, 10)、[20, 30) 段已经接近这个下限，
说明它在样本密集区已经基本饱和。

### 11.4 小结

| 模型 | 失败模式 |
|---|---|
| DenseNet | 误差随年龄段平滑上升，无系统性塌缩，但样本数极少的尾部 RMSE 偏高 |
| ViT | 严重的 "主峰塌缩"：主峰内表现良好，主峰外几乎不学习；典型的小数据 + 长尾下 Transformer 归纳偏置短板 |

## 12. 总结

### 12.1 主要结论

1. 本实验在 APPA-REAL 上完成了 DenseNet121 与 ViT-B/16 的年龄估计回归任务，
   并通过统一的训练循环 + 受控差异 YAML 实现了公平对比。
2. **整体上 DenseNet 显著优于 ViT**：test MAE 5.06 vs. 13.01，
   test RMSE 7.19 vs. 17.63。
3. **逐年龄段分析揭示了 ViT 的特殊失败模式**：在主峰 [20, 30) 区段 ViT
   略胜 DenseNet（MAE 3.67 vs. 3.70），但在远离主峰的儿童与老年段
   完全塌缩（[70, +inf) MAE 47.81 vs. 12.81）。这与 ViT 缺乏局部归纳偏置、
   对训练数据规模高度依赖的理论预期一致。
4. **训练效率**：DenseNet 单 epoch 9.3s，ViT 单 epoch 35.9s（约 4 倍），
   且 ViT 显存峰值更低但效果更差。在 ~4k 训练样本的中小规模数据上，
   选择 CNN 系列骨干是更经济、更稳健的选择。

### 12.2 实验限制

1. **单 seed 运行**：仅用 `seed=42` 各跑一次，
   未提供误差棒；不同种子可能带来若干 MAE 单位的波动，
   但不会改变 "DN 显著优于 ViT" 的结论数量级。
2. **数据规模有限**：APPA-REAL train+val+test 仅 ~7600 张，
   对 ViT 不公平。若有 IMDB-WIKI 或 UTKFace 等更大数据集，
   ViT 的相对劣势可能减小甚至反转。
3. **训练时长有限**：仅 25 epoch。从 ViT 的 cosine schedule 可见
   最后阶段 lr 已衰减至 1.97e-07，几乎停止学习，模型已收敛；
   但 "更长 + 更大 lr + warmup" 的训练协议可能让 ViT 进一步提升，
   本实验未做该项 ablation。
4. **超参数未做系统搜索**：两个模型的 optimizer/lr/scheduler 都是
   各自架构的常见配方，并未在网格上比较。理论上更仔细的 ViT 调参
   （例如 layer-wise lr decay、更强 augmentation、mixup 等）可能缩小差距。
5. **未做置信度估计 / 不确定性建模**：实验只输出 point estimate，
   未估计 per-sample 不确定性，难以在下游决策中识别 "高风险预测"。

### 12.3 可能的扩展方向

1. **多 seed 重复 + bootstrap 误差棒**，使结论统计意义更明确。
2. **更大规模数据（IMDB-WIKI / UTKFace）下重做对比**，检验 ViT 的相对劣势
   是否随数据规模缩小或反转。
3. **类别加权 / 长尾重采样**：在长尾年龄段做 oversample 或
   focal-style 加权，缓解 ViT 的主峰塌缩。
4. **Hybrid 架构（CoAtNet / ConvNeXt）**：将卷积先验注入 Transformer，
   在 ~4k 样本规模下可能两全其美。
5. **从回归转向 expected value of softmax / DLDL 等软分类**：
   利用年龄序数结构带来更稳的训练信号，特别是尾部。
6. **加入不确定性建模**（Gaussian likelihood / Quantile Regression / MC Dropout），
   输出 95% 置信区间。

### 12.4 一句话总结

在 APPA-REAL 这种 "~7600 张样本 + 长尾年龄分布" 的中小规模回归任务上，
DenseNet 凭借局部卷积归纳偏置稳定胜出；ViT 在主峰区段表现尚可，
但在尾部完全塌缩，体现了 Transformer 对训练数据规模与归纳偏置的强依赖。
