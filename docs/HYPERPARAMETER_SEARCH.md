# 道路编码器与完整模型的超参数选取

用两个 shell 入口分别进行道路编码器（表中 RMPNet）和完整 RCG-MTAD 模型的选参。
两者读取同一套已有四状态数据与固定划分。优化器固定为 AdamW。

| 超参数 | 表中范围 | 默认离散候选 | RMPNet 表中配置 | RCG-MTAD 表中配置 |
| --- | --- | --- | --- | --- |
| Epochs | 50–200 | 50, 100, 150, 200 | 150 | 150 |
| Batch size | 32–128 | 32, 64, 128 | 64 | 128 |
| Weight decay | 1e-5–1e-3 | 1e-5, 1e-4, 1e-3 | 1e-4 | 1e-4 |
| Initial LR | 1e-5–1e-3 | 1e-5, 1e-4, 1e-3 | 1e-4 | 1e-3 |
| Min LR | 0–1e-5 | 0, 1e-6, 1e-5 | 1e-6 | 1e-5 |

默认空间共 324 组离散组合。默认使用固定搜索种子从中抽取 30 组无重复配置，
并优先包含表中对应模型的配置。表中配置只是基准候选，实际最优值由验证结果决定。
自定义候选仍须位于表中范围内；若自定义空间不含表中配置，则不额外插入该配置。

## 直接通过 sh 运行

先在目标 CPU/CUDA 环境安装匹配的 PyTorch、torchvision 及项目依赖。
以下命令在 RCG-MTAD 项目目录执行。

```bash
# 1. 道路编码器选参：默认 30 组，结果保存到 outputs/hyperparameter_search/road。
DATA_DIR=/path/to/dataset \
  bash scripts/search_road_encoder.sh

# 2. 完整模型选参：自动读取上一阶段的 best.json，所有试验使用相同道路初始化。
DATA_DIR=/path/to/dataset \
  bash scripts/search_model.sh
```

默认数据根目录为项目内的 `Datasets`，可省略 `DATA_DIR`。
目录采用 `{train,valid,test}/{pic,sensor}`，也兼容原项目的 images 图片目录。
CSV 的 image_path 优先，缺失时沿用原项目按道路、时间和源样本分组的配对规则。
搜索仅加载 train 和 valid；test 留待最终评估。

若已有道路权重，完整模型选参可直接指定：

```bash
DATA_DIR=/path/to/dataset ROAD_CHECKPOINT=/path/to/road_encoder.pt \
  bash scripts/search_model.sh
```

若需要完整模型从头训练，设置 `TRAIN_FROM_SCRATCH=1`。
自定义道路选参输出目录后，第二阶段用 `ROAD_SEARCH_RESULT=/path/to/best.json`
指定对应结果。`INITIAL_TIRE_CHECKPOINT` 可为所有完整模型试验指定相同轮胎初始化。

## 搜索范围与预算

```bash
# 遍历默认的全部 324 组；grid 模式忽略 TRIALS。
DATA_DIR=/path/to/dataset SEARCH_MODE=grid \
  bash scripts/search_road_encoder.sh

# 将完整模型预算改为 50 组，并使用独立输出目录。
DATA_DIR=/path/to/dataset TRIALS=50 OUTPUT_DIR=outputs/model_search_50 \
  bash scripts/search_model.sh

# 在表中范围内自定义候选。
DATA_DIR=/path/to/dataset EPOCHS_GRID=100,150 BATCH_SIZE_GRID=64,128 \
  WEIGHT_DECAY_GRID=1e-5,1e-4 INITIAL_LR_GRID=1e-4,1e-3 MIN_LR_GRID=1e-6,1e-5 \
  TRIALS=16 OUTPUT_DIR=outputs/road_search_custom \
  bash scripts/search_road_encoder.sh

# 只写计划、打印命令；不加载模型或开始训练。使用独立预览目录。
DATA_DIR=/path/to/dataset DRY_RUN=1 OUTPUT_DIR=outputs/road_search_preview \
  bash scripts/search_road_encoder.sh

# 完整模型也可预览命令；预览模式不会加载指定的道路权重文件。
DATA_DIR=/path/to/dataset ROAD_CHECKPOINT=/path/to/road_encoder.pt \
  DRY_RUN=1 OUTPUT_DIR=outputs/model_search_preview \
  bash scripts/search_model.sh
```

常用环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `SEARCH_MODE` | `random` | `random` 抽取或 `grid` 遍历离散空间 |
| `TRIALS` | `30` | 随机模式配置数，超过空间大小时使用全部配置 |
| `SEARCH_SEED` | `42` | 控制随机配置抽取 |
| `SEED` | `42` | 所有候选使用相同训练种子 |
| `OUTPUT_DIR` | 按阶段分别保存 | 选参输出目录 |
| `CONFIG` | `configs/rcg_mtad.json` | 固定的模型与输入配置 |
| `DEVICE` / `NUM_WORKERS` | `auto` / `0` | 设备和数据加载进程数 |
| `RESUME` | `1` | 跳过同一搜索计划中已完成的试验 |
| `CONTINUE_ON_ERROR` | `1` | 记录失败并继续其他候选，设为 `0` 则立即停止 |
| `INCLUDE_REFERENCE` | `1` | 优先包含表中配置，设为 `0` 不作优先安排 |
| `FREEZE_ROAD_ENCODER` | `0` | 完整模型默认微调道路分支，设为 `1` 可固定道路分支 |
| `ROAD_LOSS_WEIGHT` | `1.0` | 固定的完整模型道路辅助损失权重 |

修改范围、模型配置、初始化、训练种子、源码或数据版本时，应使用新的输出目录。
恢复只适用于相同搜索计划；每个试验从相同初始化规则重新训练，不续接中断的优化器。

## 学习率与评选规则

搜索调用 `--lr-scheduler cosine --min-lr ...`，将学习率从候选 Initial LR
降至候选 Min LR。完整模型的道路、轮胎参数组采用同一候选初始学习率和最小学习率，
对应表中的单一完整模型学习率设置。普通训练入口仍默认使用原有的固定学习率。

Epochs 是每个候选的完整训练预算。搜索将 patience 设为该预算，避免提前停止改变
轮数搜索的含义，同时保留预算内验证表现最好的检查点。

道路编码器按验证集道路 Macro-F1 选取，完整模型按验证集轮胎四分类 Macro-F1 选取；
同分时依次比较验证总损失和试验编号。所有试验均传入 `--no-evaluate-test`，
不计算测试指标，不用测试结果进行排序。

## 输出

```text
outputs/hyperparameter_search/road/       # 完整模型对应 joint/
├── search_manifest.json                 # 候选参数、完整命令及固定实验配置
├── results.csv                          # 已完成/失败试验及验证指标，成功试验按排名排列
├── best.json                            # 最佳参数、验证分数和检查点绝对路径
└── trial_0001/
    ├── train.log
    ├── metrics.json                     # 训练/验证历史，包含每轮实际学习率
    └── road_encoder.pt                  # 完整模型对应 rcg_mtad.pt
```

最终检查点保留在其试验目录内；`best.json` 指向该文件。完整模型搜索自动读取道路
搜索的最佳检查点。独立测试集留待选参确定后进行最终评估。

本次仅编写脚本及训练接口，并进行静态检查，未执行训练或超参数搜索。
