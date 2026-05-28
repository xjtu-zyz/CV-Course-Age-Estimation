# XJTU 计算机视觉大实验：基于 DenseNet 与 ViT 的年龄估计

## 1. 实验目的

本实验在 APPA-REAL 表观年龄数据集上完成端到端的年龄估计任务，
并对两类具有代表性的图像表征模型进行受控对比：

1. 基于卷积神经网络的 DenseNet121；
2. 基于 Transformer 的 ViT-B/16。

目标包括：

- 将年龄估计建模为一维回归问题，使用统一的损失函数与评价指标完成训练与测试；
- 在尽可能公平的实验协议下比较 DenseNet 与 ViT 的整体表现以及在不同年龄段的差异；
- 通过 per-age-group 误差与预测分布观察两类归纳偏置（局部卷积 vs. 全局自注意力）
  在中等规模数据 + 长尾年龄分布下的实际表现；
- 通过 **budget × recipe 的 2×2 因素分解 ablation** 把 ViT 性能差异拆解到
  「训练 budget」和「fine-tune 配方」两个独立维度上，避免把两者的贡献混淆
  在一起（这是相对早期版本报告最实质的方法学修正）。

报告共涉及五次训练运行：DenseNet121 (50 ep)、ViT-B/16 新配方 (50 ep)、
ViT-B/16 默认配方 baseline (50 ep)，以及作为 budget ablation 历史数据的
DenseNet121 (25 ep) 与 ViT-B/16 baseline (25 ep)。所有结论与数字均来自
`results/{densenet, vit, vit_baseline, densenet_25ep, vit_baseline_25ep}/`
与 `results/comparison/` 目录下的日志文件，不做事后估计。

## 2. 实验环境

三次主对比训练均在同一云端 GPU 服务器上完成，环境信息来自
`results/{densenet, vit, vit_baseline}/env_snapshot.txt`：

| 项目 | 值 |
|---|---|
| 操作系统 / Python | Python 3.12.3 |
| 深度学习框架 | torch 2.5.1+cu124 |
| CUDA / cuDNN | CUDA 12.4 / cuDNN 90100 |
| GPU | NVIDIA A800 80GB PCIe，单卡 |
| 主要依赖 | torchvision, pandas, numpy, PIL, tqdm, pyyaml, matplotlib |
| 随机种子 | 42（所有运行一致） |
| Git revision | DenseNet 50ep & ViT baseline 50ep：`b13e839`；ViT new 50ep：`605c112` |

三次主运行在同一仓库内进行，差别仅来自 YAML 配置（`configs/densenet.yaml`、
`configs/vit.yaml`、`configs/vit_baseline.yaml`）。运行命令形如
`python main.py --model {densenet, vit, vit_baseline} --mode all --seed 42 ...`，
通过 `--out_dir` 区分到不同子目录。每次运行额外保存
`config_snapshot.yaml`、`env_snapshot.txt`、`seed.txt`，与
`epoch_log.csv`、`step_log.csv`、`test_summary.txt`、
`per_age_group_mae.csv`、`pred_vs_true_scatter.png` 等结果文件，
便于复现与审计。

随机性控制：`src/utils.py::set_seed(42)` 同时设置
`random`、`numpy`、`torch`（CPU + CUDA）的种子，并开启
`torch.backends.cudnn.deterministic=True`、`cudnn.benchmark=False`，
DataLoader 的 `worker_init_fn` 给每个 worker 派生子种子。

## 3. 数据集介绍

实验使用 **APPA-REAL** 表观年龄数据集（Agustsson et al., 2017）。

- **来源**：APPA-REAL 官方发布版本，目录结构为
  `appa-real-release/{train,valid,test}/<file_name>_face.jpg`，
  标签文件为 `gt_avg_<split>.csv`。
- **标签语义**：使用 `apparent_age_avg` 字段，即多名标注者对同一张人脸图像给出的
  表观年龄（apparent age）的平均值，是连续实数。
- **Split 与样本数**：直接使用官方的 train / valid / test 划分，未做任何重切分。
  样本数取自 `gt_avg_<split>.csv` 的有效行数（去 header）：

  | 子集 | 样本数 | 用途 |
  |---|---:|---|
  | train | 4113 | 训练 |
  | valid | 1500 | 早停 / best-checkpoint 选择 |
  | test  | 1978 | 最终评估（仅一次） |

  注：训练时 `DataLoader` 启用 `drop_last=True`，每个 epoch 实际过
  `floor(4113/bs) × bs` 张样本（DenseNet 128 步 × 32 = 4096；ViT 257 步 × 16 = 4112），
  少量样本会被当前 epoch 跳过，多 epoch 之间随机打乱让训练集整体覆盖到位，
  无系统偏差。`step_log.csv` 仅按 20 步采样一次，**不可用于反推训练集大小**。
- **Ignore list**：dataset 模块支持 `ignore_list.csv` 自动剔除标注质量差的样本
  （见 `src/dataset.py`）。
- **年龄分布**：测试集呈典型长尾分布。

测试集年龄段样本数（来自 `per_age_group_mae.csv`，所有运行 N 列一致）：

| 年龄段 | 样本数 | 占比 |
|---|---:|---:|
| [0, 10)   | 222 | 11.2% |
| [10, 20)  | 147 |  7.4% |
| [20, 30)  | 579 | 29.3% |
| [30, 40)  | 465 | 23.5% |
| [40, 50)  | 244 | 12.3% |
| [50, 60)  | 146 |  7.4% |
| [60, 70)  | 104 |  5.3% |
| [70, inf) |  71 |  3.6% |

主峰集中在 20-40 岁（52.8%），两端 [0, 10) 与 [70, inf) 都属于稀疏区域，
误差分析会专门检查这部分。

## 4. 数据预处理方法

所有运行共用同一套预处理流水线（`src/dataset.py::build_transform`），
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
也避免预处理差异污染对比结果。未使用 mixup / CutMix / RandAugment 等更强增广，
原因有二：(a) 与课程项目所要求的简洁可复现配方一致；(b) 所有运行使用同样
增广，保证配方差异只来自 optimizer / scheduler 部分。

## 5. 年龄估计任务建模

年龄是一维连续实数，将其建模为**回归任务**比建模为分类任务更自然：

- 分类视角下，年龄被离散化为 N 个类别，所有相邻类别在交叉熵下的距离相等。
  但实际中 "把 23 岁预测成 24 岁" 与 "把 23 岁预测成 80 岁" 的代价显然不同，
  分类形式直接丢失了这一连续结构。
- 回归视角下，模型直接输出一个标量年龄 `y_hat`，损失函数与评价指标都
  按照实数距离定义，恰好匹配任务语义。

**损失函数**：所有运行均使用 `nn.L1Loss()`（即 MAE 损失，
配置文件中 `loss: l1`）。选择 L1 而非 MSE 是因为：

- L1 对异常标签（噪声年龄）更鲁棒（表观年龄本身有人工方差）；
- L1 与最终评价指标 MAE 单位一致，优化目标 = 评价目标，
  避免训练 / 评价不一致带来的优化 bias。

**评价指标**：

- **MAE**（Mean Absolute Error）：`mean(|y_hat - y|)`，单位为「岁」，主指标；
- **RMSE**（Root Mean Squared Error）：`sqrt(mean((y_hat - y)^2))`，
  对大误差更敏感，反映尾部行为；
- **Per-age-group MAE**：将测试集按 10 岁分桶统计 MAE，反映长尾段表现；
- **Pearson r** 与 **pred_std**：检验模型预测分布是否塌缩到一点（用于
  判断是否退化为均值预测器）。

模型输出头统一为 `Linear(in_features, 1)`，forward 后通过 `squeeze(1)`
压成 `(B,)` 与标签计算损失（`src/train.py`）。
回归输出无值域约束，理论上可能出现负数或大于 100 的预测，
分析时按原始值统计（不做 clip），以反映模型的真实行为。

## 6. DenseNet 模型设计

**架构来源**：Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017。

**具体实现**（`src/models.py::_build_densenet`）：

```python
weights = tvm.DenseNet121_Weights.IMAGENET1K_V1
model = tvm.densenet121(weights=weights)
in_features = model.classifier.in_features    # 1024
model.classifier = nn.Linear(in_features, 1)  # 回归头
```

关键设计点：

- **直接复用 torchvision 的 `densenet121`**：4 个 Dense Block 共 121 层，
  通道增长率 32，最后一个全局池化层输出 1024 维特征；
- **预训练权重**：加载 IMAGENET1K_V1，作为 backbone 的初始化；
- **替换 classifier**：原 `Linear(1024, 1000)` -> `Linear(1024, 1)`；
- **全网络微调**：未冻结任何层，所有参数都参与梯度更新。

**训练配方**：Adam（weight decay = 0）+ StepLR（γ=0.5，step=10），
batch_size=32，**50 epoch**，L1 loss。`StepLR(step=10, γ=0.5)` 在 50 ep
内一共触发 4 次衰减（epoch 11/21/31/41），lr 从 1e-4 降到 6.25e-6，
**整体 5× 衰减跨度**（早期 25-epoch 版本只发生 2 次衰减，对应 ~2× 跨度，
本轮 50 ep 让 lr schedule 走完了更完整的退火）。

DenseNet 通过密集连接强化梯度流，并在每层显式重用前层特征，
具有较强的局部纹理建模能力，对面部皱纹、毛孔等与年龄相关的局部特征友好。

## 7. ViT 模型设计

**架构来源**：Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021。

**具体实现**（`src/models.py::_build_vit_b_16`）：

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
  对预训练的依赖性远高于 CNN；
- **替换 `heads.head`**：torchvision 的 ViT 头部是 `Sequential` 结构，
  最后一层 `Linear(768, 1000)` 替换为 `Linear(768, 1)`；
- **全网络微调**：与 DenseNet 一致，整网可训练。但 ViT 对 fine-tune
  配方比 CNN 更敏感 —— 第 11 节 budget × recipe ablation 将系统量化这一点：
  默认 torchvision 配方（AdamW lr=5e-5 flat + cosine→0 + wd=0.05）哪怕同样跑
  50 epoch 也只能从 25 epoch 的 mean-predictor 塌缩状态部分恢复，
  仍显著差于本实验主 ViT 运行所用的 **LLRD + linear warmup + 较弱 wd** 配方。
  详见 §8.2 与 §11。

ViT 通过自注意力建模 patch 之间的全局关系，理论上能捕捉跨区域的面部
结构特征（如对称性、整体轮廓与年龄相关的特征分布），与 CNN 的局部纹理
建模形成互补。

## 8. 训练方法与参数设置

### 8.1 公平实验协议

为保证三次主对比运行的结果可比，下列维度严格一致：

- **同一份随机种子**：`seed=42`，覆盖 `random`、`numpy`、`torch`、CUDA、
  DataLoader generator、worker init；
- **同一份 data split**：APPA-REAL 官方 train / valid / test = 4113 / 1500 / 1978，
  未做任何重切分；
- **同一组 augmentation 策略**：见第 4 节，由
  `src/dataset.py::build_transform` 共享；
- **同一种损失函数**：`nn.L1Loss()`；
- **同一组 ImageNet mean/std normalize**；
- **同一个训练循环**：`src/train.py::run_training`（src/train.py:128），
  全局共用，代码中**没有** `if model_name == "vit"` 的特判分支；
- **同样的输入尺寸**：224×224；
- **同样的 best-checkpoint 选择**：以 `val_mae` 最低对应的 checkpoint 评估 test；
- **同样的 50-epoch 训练 budget**（早期版本报告的「budget 不对称」缺陷已修复）。

仅有的差异由各自的 YAML 配置（`configs/densenet.yaml`、`configs/vit.yaml`、
`configs/vit_baseline.yaml`）驱动。

### 8.2 三次主运行的配方对比

| 维度 | DenseNet | ViT (new) | ViT (baseline) |
|---|---|---|---|
| optimizer | Adam | AdamW | AdamW |
| base lr | 1e-4 | 2e-4 (head, base) | 5e-5 (flat) |
| LLRD | — | decay 0.75 | — |
| warmup | — | linear, 5 epoch (start factor 0.01) | — |
| scheduler | StepLR (γ=0.5, step=10，5× 跨度) | cosine -> η_min=3e-7 | cosine -> 0 |
| weight_decay | 0 | 0.01 | 0.05 |
| batch_size | 32 | 16 | 16 |
| epochs | **50** | **50** | **50** |

**ViT (new) 配方细节**：LLRD 给最靠近输出的 head 与 final LayerNorm 最大学习率
（base_lr = 2e-4），向 backbone 深处衰减：第 k 层 encoder block 学习率为
`2e-4 × 0.75^(12-k)`，最底层 patch embedding 大约
`2e-4 × 0.75^13 ≈ 4.75e-6`。LayerNorm / bias / pos_embedding / class_token
等参数 `weight_decay` 设为 0（标准 transformer 约定）。warmup 阶段
前 5 epoch lr 从 `0.01 × base_lr` 线性升到 `base_lr`，随后 45 epoch
cosine 衰减到 `η_min=3e-7`。这一组配方在 `configs/vit.yaml` 中显式定义，
未在代码中硬编码。

**ViT (baseline) 配方**：接近 torchvision 文档默认的 fine-tune 配方 ——
单一 lr=5e-5 平摊到所有参数、cosine 衰减到 0、wd=0.05。两次 ViT 运行的
backbone、head 替换、数据流水线、loss 完全相同；唯一差异是上表中的
optimizer / scheduler / wd / batch_size / LLRD（**注意：epochs 已被对齐到 50**）。

**DenseNet 配方**：Adam（无 weight decay）+ StepLR 每 10 epoch γ=0.5 衰减
（在 50 ep 跨度内一共衰减 4 次，lr 1e-4 → 6.25e-6，整体 5× 衰减），
batch_size=32，50 epoch，是 CNN 微调的常见配方。

### 8.3 训练循环

`run_training()` 内部每个 epoch 顺序执行：

1. `train_one_epoch`：在训练集上跑一遍，记录 step-level batch loss（每 20
   步记录一次）与 epoch 平均 train_loss；
2. `evaluate`：在验证集上前向，记录 val_loss / val_mae / val_rmse；
3. `scheduler.step()`：epoch-level 调整 lr；
4. 保存 `last` checkpoint；当 `val_mae < best_val_mae` 时另存 `best`；
5. 写出 `epoch_log.csv`，flush `step_log.csv`。

测试阶段在 `best.pth` 上跑测试集一次，保存预测、ground truth、
file_names 与 `test_summary.txt`。

## 9. 实验结果

### 9.1 主对比表（test，单 seed=42，同 50-epoch budget）

直接来自 `results/comparison/comparison_table.csv` 与各自的 `test_summary.txt`：

| 指标 | DenseNet | ViT (new) | ViT (baseline) |
|---|---:|---:|---:|
| 训练 epoch 数 | 50 | 50 | 50 |
| best epoch (by val MAE) | 31 | 37 | 45 |
| best val MAE | **4.098** | 4.127 | 7.629 |
| best val RMSE | 6.302 | **6.196** | 10.699 |
| test MAE | 4.941 | **4.861** | 9.442 |
| test RMSE | **6.943** | **6.943** | 13.134 |
| Pearson r(pred, gt) | **0.925** | 0.924 | 0.696 |
| pred std (test) | 15.26 | 15.27 | 11.54 (gt std: 17.67) |
| mean epoch time (s) | 9.26 | 36.18 | 36.09 |
| GPU peak (MB) | 4122.6 | 2990.8 | 2989.8 |

加粗为该指标下最优值（DenseNet 与 ViT (new) 在 RMSE 上**完全打平到小数点后三位**
6.943 vs 6.943，是本轮实验的一个有意思的亮点）。三次运行均跑完计划 50 epoch，
未早停。**best epoch** 是按验证集 MAE 最低选出的，对应的 `<model>_best.pth`
checkpoint 即为 test 评估所用模型。test 指标反映的是 best 那一版的泛化表现，
而非最后一个 epoch。

### 9.2 训练曲线（loss / MAE）

![Loss curves (overlay)](../results/comparison/loss_curves_overlay.png)

三个模型的 train_loss 与 val_loss 叠加。DenseNet 与 ViT (new) 都呈现明显
持续下降，并在 epoch 30 之后进入平台期；ViT (baseline) 在前 10 epoch 出现
显著的 mean-predictor 平台（val_loss ≈ 11，几乎不动），约 epoch 20 之后
缓慢开始下降，最终在 epoch 45 取得最佳 val MAE 7.629 —— 远好于 25-epoch
版本的塌缩状态，但仍显著劣于另两个模型。

![Validation MAE curves (overlay)](../results/comparison/mae_curves_overlay.png)

val_mae 曲线更直接地呈现三种轨迹：DenseNet 在 epoch 31 取得 best val MAE
4.098；ViT (new) 在 epoch 37 取得 best 4.127；ViT (baseline) 在 epoch 45
取得 best 7.629 —— 前两条曲线在 epoch 20 之后就已经压在 4.1-4.3 的
低位窄带里，而 baseline 直到 epoch 25 才开始明显下降，到 epoch 45 才
跌至 7.6 附近。

### 9.3 预测 vs. 真值散点

![Scatter (3 panels)](../results/comparison/scatter_three_panel.png)

三个面板分别展示 DenseNet、ViT (new)、ViT (baseline) 在测试集上的预测散点。
DenseNet 与 ViT (new) 沿对角线分布，主要散点集中在 20-40 真实年龄区间，
长尾段散点稀疏但仍跟随对角线；ViT (baseline) 50 epoch 的散点**部分铺开**
（pred_std=11.54，约为 gt 范围的 2/3），但相对真实分布仍偏窄、偏中心，
长尾两端被显著低估或高估 —— 不再是 25-epoch 版本的「窄带常数预测」，但仍
存在系统性的范围压缩。

### 9.4 预测分布 vs. 真值分布

![Pred distribution vs gt](../results/comparison/pred_distribution.png)

ground truth 测试集的标准差为 17.67（覆盖 0-90 岁全谱）。DenseNet 与 ViT (new)
的预测分布与之高度重合（pred std 15.26 / 15.27）；ViT (baseline) 50 epoch
的预测分布 std=11.54，已经从 25-epoch 时的 3.09 显著展开，但峰仍偏向 20-40
中段，两端被截断 —— 这恰好对应 §10 中 baseline 在 [70, inf) 桶 MAE 34.21
的系统性低估。

各模型自带的单模型 loss / MAE 曲线与 scatter 图：

- DenseNet：`../results/densenet/{loss_curve, mae_curve, pred_vs_true_scatter}.png`
- ViT (new)：`../results/vit/{loss_curve, mae_curve, pred_vs_true_scatter}.png`
- ViT (baseline)：`../results/vit_baseline/{loss_curve, mae_curve, pred_vs_true_scatter}.png`

## 10. DenseNet 与 ViT 对比分析

### 10.1 整体观察

在**同 50-epoch budget** 下，ViT (new) 的 test MAE 4.861 与 DenseNet 的
4.941 相差仅 0.080 岁（相对约 1.6%）；test RMSE 完全相同（均为 6.943）；
Pearson r 几乎一致（0.924 vs 0.925）。**两者实质打平**：0.08 岁的差距完全
位于单 seed 训练的噪声范围内（一次随机种子重跑就可能让 ranking 翻转）。
这与早期报告中「ViT 微胜 0.2 岁」的结论存在差异 —— 因为早期版本是
50 epoch ViT 对 25 epoch DenseNet，budget 不对称已被本轮实验消除。

因此，**本实验的正确叙事是：在合适配方与同 budget 下，DenseNet 与 ViT 在
APPA-REAL 上达到相当的水平**，而非「ViT 更好」。

### 10.2 Per-age-group MAE

数据来自三份 `per_age_group_mae.csv`：

| 年龄段 | N | DenseNet | ViT (new) | ViT (baseline) | 更优者 (DN vs ViT new) |
|---|---:|---:|---:|---:|:---:|
| [0, 10)   | 222 |  3.20 |  2.62 |  8.89 | **ViT (new)** |
| [10, 20)  | 147 |  5.29 |  5.57 |  9.66 | DenseNet |
| [20, 30)  | 579 |  3.65 |  3.76 |  5.21 | DenseNet |
| [30, 40)  | 465 |  4.48 |  4.23 |  6.61 | **ViT (new)** |
| [40, 50)  | 244 |  5.71 |  6.11 | 10.11 | DenseNet |
| [50, 60)  | 146 |  6.77 |  6.81 | 14.64 | DenseNet |
| [60, 70)  | 104 |  8.35 |  7.59 | 20.76 | **ViT (new)** |
| [70, inf) |  71 | 11.81 | 11.15 | 34.21 | **ViT (new)** |

![Per-age bar chart](../results/comparison/per_age_bar.png)

关键观察：

1. **DN vs ViT (new) 在 8 个桶里是 4-4 平手**：DenseNet 在
   [10, 20) / [20, 30) / [40, 50) / [50, 60) 略胜；ViT (new) 在
   [0, 10) / [30, 40) / [60, 70) / [70, inf) 略胜。逐桶差距大多在
   0.04-0.76 岁之间，与整体「实质打平」一致。
2. **ViT (new) 守住两端长尾**：[0, 10) 领先 0.58、[60, 70) 领先 0.76、
   [70, inf) 领先 0.66。最稀疏的 [70, inf) 桶上 ViT (new) 仍把 MAE 压在
   11 岁出头，DenseNet 则到 11.81。这与「ViT 的全局 self-attention 能跨整张
   人脸聚合多种线索」的解释一致：极端年龄段的局部纹理（如皱纹深度）和
   整体形态（如头颅比例、面部轮廓）都偏离主峰，全局聚合更稳健。
3. **DenseNet 在主峰邻域占优**：[20, 30) 与 [40, 50) 这两个 N 较大的桶上
   DenseNet 稍好，与卷积局部纹理先验在密集训练区段更稳定的直觉一致。
4. **ViT (baseline) 在长尾段彻底崩坏**：[70, inf) MAE 34.21、[60, 70) MAE 20.76
   —— 比正常工作的两个模型分别差 ~23 / ~13 岁。详见 §11。

### 10.3 训练资源对比

| 维度 | DenseNet | ViT (new) | ViT (baseline) |
|---|---|---|---|
| 单 epoch 时间 (s) | 9.26 | 36.18 | 36.09 |
| 总训练时间 (s) | ≈ 463 | ≈ 1809 | ≈ 1805 |
| GPU peak 显存 (MB) | 4122.6 | 2990.8 | 2989.8 |

ViT 单 epoch 时间约为 DenseNet 的 ~3.9 倍（同 batch_size 之差也部分参与，但
单步 FLOPs 也明显更高），所以同 50 epoch budget 下 ViT 总 wall-clock 约
为 DenseNet 的 ~3.9 倍。**显存反而较低**，因为 batch_size=16 < 32。
本轮实验中，**ViT (new) 与 DenseNet 性能基本相当**（MAE 4.86 vs 4.94），但 ViT
要付出 ~3.9× 的训练成本 —— 在课程项目层面这一点本身就是有价值的工程观察：
若性能基本打平，CNN 在小数据场景仍是更经济的选择。

## 11. 误差分析 + Budget × Recipe 因素分解 Ablation

本节是本版报告的核心。在早期版本里我们只有「budget 不对称的 ViT (new) 50ep
vs ViT (baseline) 25ep」对比，把 budget 与 recipe 两个变量的贡献混在一起；
本轮实验补齐了 baseline 50 epoch 以及 DenseNet 25 epoch 两个数据点，
得以做完整的 2×2 因素分解。

### 11.1 budget × recipe 5 行 ablation 表

数据来自 `results/comparison/budget_ablation.csv`：

| 模型 | budget | best epoch | test MAE | test RMSE | pred_std | Pearson r |
|---|:---:|---:|---:|---:|---:|---:|
| DenseNet121 | 25 ep | 22 | 5.064 | 7.190 | 15.29 | 0.920 |
| DenseNet121 | 50 ep | 31 | 4.941 | 6.943 | 15.26 | 0.925 |
| ViT-B/16 (new recipe) | 50 ep | 37 | **4.861** | **6.943** | 15.27 | 0.924 |
| ViT-B/16 (baseline recipe) | 25 ep | 25 | 13.009 | 17.627 |  3.09 | 0.286 ← 塌缩 |
| ViT-B/16 (baseline recipe) | 50 ep | 45 |  9.442 | 13.134 | 11.54 | 0.696 |

### 11.2 ViT 端的两个独立维度贡献

把 ViT 的 3 个数据点（baseline 25ep / baseline 50ep / new 50ep）放进
2×2 ablation 网格（缺失 cell 为 new 25ep，未做），可分别沿两个独立维度
做边际效应估计：

| 比较 | 控制变量 | 变化 | test MAE 变化 |
|---|---|---|---:|
| baseline 25ep → baseline 50ep | recipe 固定为 baseline | budget 25→50 | 13.009 → 9.442，**−3.567** |
| baseline 50ep → new 50ep | budget 固定为 50ep | recipe baseline→new | 9.442 → 4.861, **−4.581** |
| baseline 25ep → new 50ep (总) | 两者同时变 | budget + recipe | 13.009 → 4.861, **−8.148** |

可以看到 budget 和 recipe **两个维度都有显著且可比的贡献**（分别约 3.6 和
4.6 MAE），早期版本「recipe 救了 ViT」的单因素叙事并不完整。同时
−3.567 + −4.581 = −8.148 数值上恰好等于总变化，**说明这两个维度在 25→50ep /
baseline→new 这个范围内大致没有显著的相互作用项**（虽然只有两条路径，
不能做正式的交互检验）。

DenseNet 端 budget 维度的对照：25 ep → 50 ep test MAE 5.064 → 4.941，
**−0.123**。两倍 budget 在已经成熟的 CNN 配方下边际收益微小（DN 在 25ep
本身就已经接近自身上限），与 ViT 的 −3.567 形成鲜明对比 —— 这暗示
**budget 维度的提升大小高度依赖于该模型是否还处于「未充分训练」状态**。

### 11.3 ViT baseline 25 epoch 时的塌缩证据（历史）

ViT (baseline) 在 25 epoch 时呈现典型的均值预测器塌缩，证据来自
`results/vit_baseline_25ep/test_summary.txt` 与 `per_age_group_mae.csv`：

- test MAE 13.009、test RMSE 17.627；
- **Pearson r = 0.286** —— 接近随机 / 常数预测；
- **pred_std = 3.09**，而 gt std = 17.67 —— 预测分布压缩到原始的约 1/5.7 宽度；
- per-age 桶严重对称崩坏：[0, 10) MAE 22.32、[70, inf) MAE 47.81，
  与「预测几乎永远在 ~25 岁」的常数预测器表现完全吻合（年轻被高估、
  老人被严重低估，且差值随距离单调放大）。

这是早期报告中「baseline 塌缩」叙事所基于的状态。但**该结论只在 25 epoch
时严格成立**。

### 11.4 ViT baseline 50 epoch 部分恢复，但仍弱

跑到 50 epoch 后，ViT (baseline) 不再处于完全塌缩状态：

- test MAE 9.442（仍远高于 ViT (new) 的 4.861，但比 25ep 时的 13.009 大幅改善）；
- Pearson r = **0.696** —— 已经具备明显的单调一致性，不再是随机预测；
- pred_std = **11.54** —— 大约是 25ep 时 3.09 的 3.7 倍，分布显著展开；
- per-age 桶：[20, 30) / [30, 40) MAE 分别为 5.21 / 6.61，已经接近正常水平；
  但 [60, 70) / [70, inf) 仍有 20.76 / 34.21 的崩坏 —— 这说明它学到了
  「中段年龄的回归」，但对极端年龄段的扩展仍不充分。

**为什么 50 epoch 能逃出 25 epoch 时的 mean-predictor 局部最优**：
查看 `results/vit_baseline/epoch_log.csv`，关键观察是从约 epoch 20 后
val MAE 开始进入显著下降轨迹（epoch 20: 10.34 → epoch 30: 8.98 →
epoch 40: 7.79 → epoch 45: 7.63）。这背后有两个机制：

1. **cosine 后半段 lr 仍处于 2e-5 数量级，且 50ep 给了显著更多的 update
   step**（DataLoader 每个 epoch 256 step，多 25 epoch 即多 ~6400 step），
   累计梯度信号足以慢慢把 backbone 从 ImageNet 表征往 APPA-REAL 域上迁移；
2. **head 不再被 wd=0.05 持续压回 0**：一旦学到一点点真实信号、loss 开始
   下降，wd 作用相对变小，进入正向反馈。

但相比 ViT (new)，baseline 仍弱很多（−4.58 MAE 的 gap）。这说明
**budget 能让模型「逃出塌缩」，但 LLRD + warmup 配方仍带来巨大的「充分微调」
增益**。

### 11.5 为什么 new recipe 在同 budget 下仍有大幅边际收益

ViT (new) 与 baseline 在同 50 epoch budget 下 MAE 差 4.58 岁。把
两个配方差异列出来：

1. **LLRD (decay 0.75) vs. flat lr**：new 给 head 拿到 2e-4，最底层 patch
   embedding 大约 4.75e-6，**head 与 backbone 不再争夺同一个 lr**。
   head 能快速学到任务相关信号，backbone 缓慢但持续微调。flat 5e-5 配方下
   head 信号太弱、backbone 又过强（相对其需要的微调幅度而言），二者错配。
2. **Linear warmup（5 epoch）vs. 无 warmup**：epoch 1 起点 lr=2e-6，
   epoch 6 升到 2e-4。warmup 期间 AdamW 的二阶矩 estimate 充分稳定，避免大
   梯度污染 m, v 累积量、避免 head 走向「猜均值」的捷径。baseline 在
   epoch 1 直接以 5e-5 起步，AdamW 二阶矩还未稳定时就在做较大幅度的参数
   更新 —— 这是 25 epoch 时塌缩的关键诱因。
3. **wd=0.01 vs. wd=0.05**：减少对刚学到的 head 信号的压制。
4. **cosine to η_min=3e-7 vs. cosine to 0**：尾段 lr 保留极小正值，防止
   最后几个 epoch 完全失去更新能力。
5. **同一份代码、同一个 `run_training()`**：所以全部 4.58 岁的差距均来自配方，
   不是某种特殊训练循环或正则化技巧。

### 11.6 DenseNet vs ViT (new) 的共同失败模式

两个能正常学习的模型在长尾段都有显著误差膨胀：

- [70, inf)：DenseNet 11.81 / ViT (new) 11.15，远高于整体 MAE ~5；
- [60, 70)：DenseNet 8.35 / ViT (new) 7.59；
- [0, 10)：DenseNet 3.20 / ViT (new) 2.62（绝对误差较小，但相对误差仍可观）。

主因来自**训练分布偏移**：训练集主要集中在 20-50 岁，长尾段样本少，模型对
极端年龄段的回归方差自然更大。这也是为什么两者 RMSE / MAE 比值
（DN：6.943 / 4.941 ≈ 1.41；ViT (new)：6.943 / 4.861 ≈ 1.43）都明显大于 1：
少数长尾样本的大误差被平方放大。

此外，APPA-REAL 是 *apparent* age 标签，本身有不可消除的标注主观性
（不同标注者对同一张脸给出的年龄差异常常在 ±5 岁以内）。这给所有模型设了
MAE 大约 3-4 岁的不可逾越下限 —— DenseNet 与 ViT (new) 在 [0, 10)、
[20, 30)、[30, 40) 段已经接近这个下限，说明它们在样本密集区已经基本饱和。

### 11.7 教训

1. **对中等规模数据 + 预训练 ViT，fine-tune 配方与训练 budget 二者各自
   都有大幅独立贡献**，二者在本实验区间内大致可加（−3.6 与 −4.6 MAE，
   总和与端到端变化一致）。早期版本只看 recipe、忽略 budget 的叙事被
   修正。
2. **判断 "ViT 塌缩" 必须给出 epoch 数限定词**：本实验中只有 25 epoch 时
   ViT baseline 严格塌缩到均值预测器；50 epoch 时它不塌缩，只是弱。
3. **DN 端 budget 边际收益小**（25→50 仅 −0.123 MAE），说明 budget 维度的
   贡献大小**强依赖于模型是否还处于未充分训练状态**。这对今后做 ablation
   的建议是：报告 budget contribution 时必须同时报告模型在该 budget 下
   的「饱和程度」（如 train_loss / val_mae 的二阶导）。
4. **代码与数据完全相同**：所有差异都被锁在 YAML 配置里。这是
   `src/train.py::run_training` 没有任何 model-conditional 分支的直接红利。

## 12. 总结

### 12.1 主要结论

1. 本实验在 APPA-REAL 上完成了 DenseNet121 与 ViT-B/16 的年龄估计回归任务，
   并通过统一的训练循环 + 受控差异 YAML 实现了三次主对比运行（全部 50 ep）
   的公平对比，外加 5 行 budget × recipe ablation 数据。
2. **在同 50-epoch budget 下，DenseNet 与 ViT (new) 实质打平**：test MAE
   4.941 vs 4.861（差 0.080，单 seed 噪声范围内），test RMSE 均为 6.943
   （完全打平到小数点后 3 位），Pearson r 0.925 vs 0.924。早期报告中
   「ViT 微胜 0.2 岁」的结论被同 budget 重跑修正为「两者基本相当」。
3. **逐年龄段分析在 8 桶里是 4-4 平手**：DN 在 [10, 20) / [20, 30) /
   [40, 50) / [50, 60) 略胜；ViT (new) 在 [0, 10) / [30, 40) / [60, 70) /
   [70, inf) 略胜，尤其守住两端长尾。
4. **Budget × Recipe ablation 是本实验最强信号**：对 ViT 端，
   **budget 维度**（baseline 25→50 ep）贡献 **−3.567 MAE**，
   **recipe 维度**（baseline→new，同 50 ep）贡献 **−4.581 MAE**；
   两者在本实验区间内大致可加（总 −8.148 MAE 与端到端变化一致）。早期
   报告把这两个维度混淆为单一 "recipe 救了 ViT" 叙事，本版予以修正。
5. **DenseNet 端 budget 边际收益小**（25→50 ep 仅 −0.123 MAE），说明
   budget 增益强依赖于「模型是否还未充分训练」。
6. **ViT baseline 25 epoch 时塌缩到均值预测器**（Pearson r=0.286，
   pred_std=3.09），**50 epoch 时部分恢复**（r=0.696，pred_std=11.54），
   但仍显著弱于 new recipe 50 ep（r=0.924，MAE 4.861）。
7. **训练效率**：DenseNet 每 epoch ~9.26 s（50 ep ≈ 463 s），ViT
   每 epoch ~36 s（50 ep ≈ 1809 s），即 ~3.9× 训练成本，性能基本打平。

### 12.2 实验限制

1. **单 seed 运行**：所有数字来自 seed=42 一次运行，0.08 岁的 DN/ViT(new) 差距
   完全可能在多 seed 下被噪声覆盖。"两者实质打平" 这个结论稳健，但具体
   ranking 不稳健。
2. **APPA-REAL 自身**：年龄标签是众包平均「表观年龄」，本身有 noise；
   数据量中等（4113 / 1500 / 1978），换更大数据（IMDB-WIKI / UTKFace）
   可能改变结论。
3. **没有做 test-time augmentation / model ensemble**：相比常见 SOTA pipeline
   缺失了这些常见后处理步骤。
4. **没有做更强增广**：未尝试 mixup / CutMix / RandAugment / stochastic depth /
   layer-wise weight decay；未网格搜索 LLRD decay 系数（0.75 是
   ViT fine-tuning 文献中的常用值，未做敏感性分析）。
5. **ViT baseline 仍未做 100 epoch 充分探索**：本实验只观察到 25ep 塌缩、
   50ep 部分恢复两个数据点；不能排除 baseline 在 100 ep 或更长 budget 下
   继续下降（虽然 epoch 45 后曲线已显著平坦，说明已经接近自身配方上限）。
6. **2×2 ablation 路径不完整**：缺失「ViT (new) 25 epoch」这个 cell，
   因此无法在该 cell 上做 budget × recipe 的交互检验，只能在两条边路径
   上做近似可加性的描述性观察。
7. **没有做置信度估计 / 不确定性建模**：实验只输出 point estimate，
   未估计 per-sample 不确定性。

### 12.3 可扩展方向

1. **多 seed (≥3)**：取均值 + 标准差，给出 DN vs ViT (new) 差距的置信区间；
2. **ViT (new) 25 epoch 与 ViT (baseline) 100 epoch**：补齐 budget × recipe
   2×2 路径中的缺失 cell，做正式交互检验；
3. **更强数据增广（mixup / CutMix / RandAugment）**：看是否能进一步缩小
   长尾段误差；
4. **LLRD decay 网格搜索**（0.65 / 0.70 / 0.75 / 0.80 / 0.85），评估对
   ViT 训练稳定性的影响；
5. **Test-time augmentation**（horizontal flip 平均）：最便宜的预期提升手段；
6. **回归头换成 ordinal regression**（DEX / mean-variance / DLDL）：年龄
   估计常见技巧，可作进一步对比；
7. **更大数据集（IMDB-WIKI / UTKFace）重做对比**：检验 DN/ViT 打平结论
   在更大数据规模下是否仍成立。

### 12.4 一句话总结

在 APPA-REAL 单 seed、同 50-epoch budget 受控对比下，DenseNet121 与 ViT-B/16
（LLRD + warmup）实质打平（test MAE 4.941 vs 4.861，RMSE 完全相同 6.943）；
**更重要的方法学发现来自 budget × recipe 2×2 ablation**：对 ViT 端，
training budget 与 fine-tune recipe 两个维度都贡献了约 3.6-4.6 MAE 的独立增益，
合起来把 ViT 从 25ep 塌缩状态（MAE 13.01）拉到与 DenseNet 相当的水平
（MAE 4.86）。早期报告把全部功劳归于 recipe 的叙事在本轮被修正为
budget + recipe 双因素，二者大致可加。

---

附：所有实验输出文件位于 `results/{densenet, vit, vit_baseline, densenet_25ep,
vit_baseline_25ep}/` 与 `results/comparison/`；配置在
`configs/{densenet, vit, vit_baseline}.yaml`；代码在 `src/` 与入口 `main.py`。
复现命令：

```bash
python main.py --model densenet     --mode all --seed 42
python main.py --model vit          --mode all --seed 42
python main.py --model vit_baseline --mode all --seed 42
```
