# RCG-MTAD 合并说明

代码整合自 `Road-Conditioned-Tire-Anomaly-Detection` 的道路编码器和四分类器。
训练直接读取已有四状态数据，类别包括健康、漏气、过热和不平衡。

## 模型与训练

`RCGMTAD.forward(image, tire_acc, imu, vehicle_state)` 一次返回 `output.road`
和 `output.tire`，二者均包含 logits 和 probabilities。`zroad`、道路概率与
FiLM 均位于计算图中。道路分支用加速度前 256 点，轮胎分支用前 1024 点；
每条短窗口先独立补零，避免推理结果受同批其他样本长度影响。

道路与轮胎分类的类别顺序分别为：

- 积水路面、破损路面、水泥路面、沥青路面；
- healthy、tire_leak、tire_overheat、tire_imbalance。

图像由数据接口缩放并执行 ImageNet 标准化。模型接口的传感器是原始物理量，
车辆状态顺序是 `[pressure_kPa, temperature_C, speed_kmh]`。速度参与道路门控，
轮胎分类器的标量输入仅为胎压、胎温。默认 IMU 为 9 通道，前六通道依次为
ax、ay、az、gx、gy、gz，后三通道补零。TTSPD 的 IMU 窗口统计值保留原有
`[1,9]` 适配方式；不能将其解释为采样频率更高的原始 IMU 时序。

第一阶段独立训练道路编码器并保存其权重和标准化参数；第二阶段训练整个模型，
默认损失为 `CE_tire + road_loss_weight * CE_road`。第二阶段可加载第一阶段的
检查点，也可从头训练。`--freeze-road-encoder` 支持固定已训练道路分支；默认
对其微调。原 `train-tire-classifier` 入口已指向整体训练，不再接受缓存特征。

所有拟合仅使用 train。道路输入标准化使用健康训练行，温压曲线只使用健康
训练行，轮胎特征标准化使用训练行。道路预训练检查点自带的标准化参数在第二
阶段直接沿用。验证集用于选择最优权重，测试集仅在选择完成后评估。

## 安装和运行

先按 CPU/CUDA 平台安装匹配的 PyTorch 和 torchvision，再在仓库目录运行：

```bash
pip install -e ".[test]"

# 两阶段读取相同的 train/valid/test 文件夹，单独预训练道路编码器。
train-road-encoder --data-dir Datasets \
  --config configs/rcg_mtad.json --output-dir outputs/road_encoder

# 将当前四状态数据用于整个模型的训练，无需提前导出 zroad/Proad。
train-rcg-mtad --data-dir Datasets \
  --road-checkpoint outputs/road_encoder/road_encoder.pt \
  --config configs/rcg_mtad.json --output-dir outputs/rcg_mtad

infer-rcg-mtad --data-dir Datasets \
  --checkpoint outputs/rcg_mtad/rcg_mtad.pt --split test

pytest
```

也可使用 `python -m tire_anomaly_system.cli.train_road_encoder`、
`python -m tire_anomaly_system.cli.train_model` 和
`python -m tire_anomaly_system.cli.infer_model`，或 `scripts/` 中同名脚本。
Shell 脚本默认读取项目内的 `Datasets`，可用 `DATA_DIR` 指定其他数据根目录。

道路独立训练和整体训练使用同一套已有数据划分，最终测试集在两个阶段均留出。

旧 RTI-MFNet 权重可通过 `--road-checkpoint` 严格加载；架构维度从权重推断，
非张量门控温度按原构造器的 1.0 恢复。旧权重不含 scaler 时，需要在当前健康
训练行上重新拟合，因此训练后的结果需要重新评估，不能直接视为论文既有结果。
旧轮胎权重可通过 `--initial-tire-checkpoint` 初始化名称及形状兼容的层；类别
顺序不一致时重置分类头，并在当前训练集重新拟合全部轮胎标准化与温压校准。

`outputs/road_encoder/road_encoder.pt` 为可独立复用的道路检查点；
`outputs/rcg_mtad/rcg_mtad.pt` 同时保存两个分支、模型配置、传感器标准化和
温压校准。推理只需后者一个文件，不需要原项目目录、独立道路权重或 scaler。
训练导出 `metrics.json` 和测试 `predictions.csv`，推理导出 `summary.json`
和 `predictions.csv`，包含道路类别/置信度、四状态类别/概率及温压物理特征。

## 已准备好的数据与 DataLoader

数据根目录名称可自定，项目内默认使用 `Datasets`。各集合事先整理为：

```text
Datasets/
├── train/
│   ├── pic/<道路类别>/*.png
│   └── sensor/**/*.csv
├── valid/
│   ├── pic/<道路类别>/*.png
│   └── sensor/**/*.csv
└── test/
    ├── pic/<道路类别>/*.png
    └── sensor/**/*.csv
```

每个集合也可沿用原项目的 `images/` 名称；`pic/` 与 `images/` 保留一个即可，
加载器自动识别，不需要移动或重命名已有图片。

`data/dataloader.py` 中的 `MultimodalDataset` 每次只读取指定集合，递归读取其
`sensor/` 内所有 CSV，每行对应一个样本。CSV 不需要 split 列，也不需要特定
文件名或 `aligned` 中间目录；目录归属是唯一的集合来源，遗留 split 列会被忽略。
文件夹内只放样本 CSV，文件清单等说明文件放到 `sensor/` 之外。

| CSV 字段 | 含义 |
| --- | --- |
| `image_path` | 优先使用；对应当前集合 `pic/` 或 `images/` 中的图片 |
| `road_label` 或 `road_type` | 0..3，或积水路面、破损路面、水泥路面、沥青路面；同时提供时应一致 |
| `anomaly_class` | 0=healthy、1=tire_leak、2=tire_overheat、3=tire_imbalance |
| `pressure_kPa`, `temperature_C`, `speed_kmh` | 原始胎压、胎温、车速 |
| `imu_ax_mean`, `imu_ay_mean`, `imu_az_mean` | IMU 三轴加速度窗口统计值 |
| `imu_gx_mean`, `imu_gy_mean`, `imu_gz_mean` | IMU 三轴角速度窗口统计值 |
| `g_0`, `g_1`, ... | 按数字后缀排列的轮胎加速度窗口 |
| `anomaly_type` | 可选；提供时必须与当前四类定义及 anomaly_class 一致 |
| `sample_id`, `source_row_id`, `group_id`, 时间列 | 元数据；没有 image_path 时，source_row_id 和时间列为配对必需字段 |

`image_path=pic/沥青路面/001.png` 相对于当前集合目录；
`image_path=沥青路面/001.png` 相对于当前集合的图片目录。路径中的 `pic/` 与
`images/` 前缀均映射到实际图片目录，也接受指向该目录内部文件的绝对路径。

没有 `image_path` 时，采用原项目 `TTSPDAlignedDataModule.match_image_sensor_data`
的有标签数据配对规则：按道路分别处理，按 trial_id 和时间排序源记录，图片按
文件名自然排序；源样本序号为 i、源样本数为 S、图片数为 I 时，使用第
`floor(i / S * I)` 张图片。相同 source_row_id 的记录共用同一张图片，保留
全部传感器行。该规则是沿用原项目的按序比例关联，不代表按采集时间精确同步。
明确提供的 image_path 始终优先；同一源样本已有明确图片时，其缺失路径副本沿用该图。

`anomaly_type` 接受 none/healthy、tire_leak、tire_overheat、tire_imbalance。
旧的 pressure_relation_anomaly、temperature_relation_anomaly、pressure_leak
数据会明确报错，不把旧编号重新解释成当前四类；训练和带标签推理均执行该检查。

两个训练阶段都要求明确的道路标签和四状态标签；训练集保留健康类用于道路
标准化与温压校准，至少需要两条健康训练记录。数据按既有记录读取，不重划分、
不增删类别、不重采样，也不构建异常数据。数据加载器保留必要的模型输入适配：
图像缩放及 ImageNet 标准化，加速度窗口截取/补零，六个 IMU 统计值按配置补零。
传感器字段中的空单元格或 NaN 在读取时统一填为 0，包括加速度窗口、IMU 和
胎压/胎温/车速。训练、验证和测试使用相同规则，保留全部记录和已有图片配对，
原始 CSV 不会被改写。无穷值、非数字内容以及缺失的必需列仍会报错。

普通训练读取 train、valid、test；选参使用 `--no-evaluate-test`，仅打开 train
和 valid。推理的 `--split` 直接选择同名目录，支持 train、valid、test、all；
默认 test，all 则读取全部三个目录。只发布验证集时可直接使用 `--split valid`，
不要求存在训练集和测试集。推理允许缺失道路和轮胎标签，缺失值保留为 -1，
并沿用检查点中的标准化参数。

```python
from tire_anomaly_system.data import MultimodalDataset, make_loader

dataset = MultimodalDataset("Datasets", "valid")
loader = make_loader(dataset, batch_size=64, workers=0)
```
