# RCG-MTAD

训练和推理直接读取 `Datasets/{train,valid,test}/{pic,sensor}` 下已准备好的四状态数据
（健康、漏气、过热、不平衡），兼容原项目的 `images` 目录。图片优先通过 CSV
的 `image_path` 对应，缺失路径时沿用原项目按道路和源样本分组的配对规则。

## 文件结构

```text
RCG-MTAD/
├── .gitattributes
├── .gitignore
├── README.md
├── pyproject.toml                         # 包配置、依赖与命令行入口
├── requirements.txt                      # 运行依赖
├── Datasets/                             # 按集合目录读取已准备好的数据
│   ├── train/                            # 训练集目录
│   ├── valid/                            # 当前提供的验证集
│   │   ├── images/                       # 路面图片，也兼容 pic/ 名称
│   │   └── sensor/                       # 按道路和试验组织的对齐传感器 CSV
│   └── test/                             # 测试集目录
├── configs/
│   └── rcg_mtad.json                      # 整体模型与输入配置
├── scripts/
│   ├── train_road_encoder.sh              # 道路编码器独立训练
│   ├── train_model.sh                     # 整体模型联合训练
│   ├── train_classifier.sh                # 原轮胎训练入口，转至整体训练
│   ├── search_road_encoder.sh             # 道路编码器超参数选取
│   ├── search_model.sh                    # 完整模型超参数选取
│   ├── search_hyperparameters.sh          # 两阶段共享选参参数与调用入口
│   ├── infer_model.sh                     # 整体模型推理
│   └── infer_classifier.sh                # 原轮胎推理入口，转至整体推理
├── src/
│   └── tire_anomaly_system/
│       ├── __init__.py
│       ├── checkpoint.py                 # 旧道路权重检查与加载
│       ├── model_checkpoint.py           # 道路预训练及整体模型检查点
│       ├── models/
│       │   ├── __init__.py
│       │   ├── rcg_mtad.py                # 道路与轮胎检测的端到端模型
│       │   ├── road_encoder.py            # 多模态道路编码器
│       │   ├── mobilenetv4_simam.py       # MobileNetV4 + SimAM 图像分支
│       │   └── tire_classifier.py         # 振动、温压与道路条件四分类器
│       ├── data/
│       │   ├── __init__.py
│       │   ├── labels.py                  # 健康、漏气、过热、不平衡的类别定义
│       │   └── dataloader.py              # 按目录配图、传感器缺值补零与 DataLoader
│       ├── training/
│       │   ├── __init__.py
│       │   └── engine.py                  # 标准化、训练、验证与预测导出
│       └── cli/
│           ├── __init__.py
│           ├── common.py                  # 共享参数、配置与数据加载
│           ├── search_hyperparameters.py  # 候选计划、顺序训练调用与验证集排序
│           ├── train_road_encoder.py      # 道路编码器独立训练入口
│           ├── train_model.py             # 整体模型训练入口
│           ├── train_tire_classifier.py   # 原训练命令兼容入口
│           ├── infer_model.py             # 单检查点端到端推理入口
│           └── infer_tire_classifier.py   # 原推理命令兼容入口
└── tests/
    ├── conftest.py                        # 测试样本与运行配置
    ├── test_end_to_end.py                 # 联合训练、梯度与检查点测试
    ├── test_road_encoder.py               # 道路编码器测试
    └── test_dataloader.py                 # 目录配图、缺值补零及批量加载测试
```
