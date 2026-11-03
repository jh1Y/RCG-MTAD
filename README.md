# RCG-MTAD

训练和推理直接读取已有四状态数据（健康、漏气、过热、不平衡）及其既定划分。

## 文件结构

```text
RCG-MTAD/
├── .gitattributes
├── .gitignore
├── README.md
├── pyproject.toml                         # 包配置、依赖与命令行入口
├── requirements.txt                      # 运行依赖
├── configs/
│   └── rcg_mtad.json                      # 整体模型与输入配置
├── docs/
│   ├── INTEGRATION.md                     # 合并说明、数据格式与运行方法
│   ├── HYPERPARAMETER_SEARCH.md           # 两阶段超参数范围、运行方法与结果说明
│   └── VALIDATION.md                      # 合并验证记录
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
│       │   ├── end_to_end.py              # 原始多模态数据接口与划分校验
│       │   ├── synchronized_samples.py    # 同步样本归档及 CSV 清单读取
│       │   └── ttspd_multimodal_data.py   # 已有四状态 TTSPD 数据与固定划分读取
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
    ├── conftest.py                        # 测试环境配置
    ├── test_end_to_end.py                 # 联合训练、梯度与检查点测试
    ├── test_road_encoder.py               # 道路编码器测试
    └── test_ttspd_raw_dataset.py          # TTSPD 原始数据与划分测试
```
