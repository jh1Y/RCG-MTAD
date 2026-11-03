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

# 两阶段使用相同的显式划分，单独预训练道路编码器。
train-road-encoder --data-dir /path/to/TTSPD --sensor-subdir sensor \
  --config configs/rcg_mtad.json --output-dir outputs/road_encoder

# 将当前四状态数据用于整个模型的训练，无需提前导出 zroad/Proad。
train-rcg-mtad --data-dir /path/to/TTSPD \
  --sensor-subdir sensor \
  --road-checkpoint outputs/road_encoder/road_encoder.pt \
  --config configs/rcg_mtad.json --output-dir outputs/rcg_mtad

infer-rcg-mtad --data-dir /path/to/TTSPD \
  --sensor-subdir sensor \
  --checkpoint outputs/rcg_mtad/rcg_mtad.pt --split test

pytest
```

也可使用 `python -m tire_anomaly_system.cli.train_road_encoder`、
`python -m tire_anomaly_system.cli.train_model` 和
`python -m tire_anomaly_system.cli.infer_model`，或 `scripts/` 中同名脚本。
Shell 脚本中的数据路径由 `DATA_DIR` 显式指定，默认项目路径从脚本位置推导。

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

## 通用原始样本接口

若不使用 TTSPD 目录，可用 `--samples samples.npz`，或 CSV 清单。NPZ 至少包含：

| 键 | 形状或含义 |
| --- | --- |
| image 或 image_path | `[N,H,W,3]` RGB uint8/0..1，或相对图像路径 |
| tire_acc | `[N,L]` 原始加速度，默认截取/补零至 1024 |
| gyro | `[N,9]`、`[N,1,9]` 或 `[N,9,1]` 原始 IMU |
| vehicle_state | `[N,3]`，kPa、摄氏度、km/h |
| road_label | `[N]`，0..3 |
| anomaly_class | `[N]`，0..3；两个训练阶段均需提供 |
| split | `[N]`，train/val/test，接受 validation 别名 |
| sample_id | `[N]` 唯一样本标识，省略时使用行号 |
| source_row_id | 可选；同一源观测窗口的标识，用于检查跨分区重叠 |

CSV 清单的传感器路径列为 `tire_acc_path`、`gyro_path`，分别指向 NPY 等数组；
图像路径列为 `image_path`；标量列为 `tire_pressure`、`tire_temperature`、`speed`。
其余元数据与 NPZ 一致。相对路径从清单目录解析，可用 `--data-root` 覆盖。
训练必须有显式 train 和 val，两个训练阶段的 train 均须含四种轮胎状态。推理数据
可省略标签和 split，此时使用 `--split all`。推理不会重新拟合模型中的 scaler。

若输入为真实 IMU 时序，可在 JSON 中统一修改 inputs.imu_length 和 gyro_indices；
分支维度会从 input/road 配置推导。模型保留原四分类器的温压和振动结构，分支
消融通过 tire 配置进行。所选 PyTorch/torchvision 版本需彼此兼容。

## 已有 TTSPD 四状态数据

默认从 `TTSPD/sensor/` 读取 `*_aligned.csv`。也可通过 `--sensor-subdir`
指定已有数据目录，保持实际目录名称不变。

每条记录的 `anomaly_class` 必须显式给出：0=healthy、1=tire_leak、
2=tire_overheat、3=tire_imbalance。训练时缺失标签会报错，不默认补为健康类。
加载器读取 `dataset_split` 或 `split` 列，也接受 `train/val/test` 父目录；
`validation` 为 `val` 的别名。列值与目录冲突时会报错，不进行随机重划分。

```text
TTSPD/
├── images/<道路类别>/*.{jpg,png,...}
└── sensor/
    ├── train/<道路类别>/<试验编号>/aligned/*_aligned.csv
    ├── validation/<道路类别>/<试验编号>/aligned/*_aligned.csv
    └── test/<道路类别>/<试验编号>/aligned/*_aligned.csv
```

已有 CSV 需要胎压、胎温、车速、六个 `imu_a*_mean`/`imu_g*_mean` 列及 `g_*`
加速度列。`road_type` 或道路目录用于确定道路类别，也可显式给出 `road_label`。
每条 CSV 记录保留一次；加载器不改变类别比例或记录数。

若已有 `image_path` 列，直接使用该路径，相对路径从 TTSPD 根目录解析。
否则保留按道路类别、源观测顺序与图像顺序配对的兼容方式；同一源观测使用同一
图像。建议在已有数据中提供明确的图像路径，以固定每条记录的图像对应关系。
单独加载测试目录和无标签部署数据不需要训练、验证数据，也不重新拟合 scaler。
