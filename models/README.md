# 酒店客服意图分类模块

基于 BGE-M3 编码器和 PyTorch 分类头，识别查询价格、下单预订、议价砍价、常规咨询四类意图。

## 文件用途

```text
models/
├── intent_config.py            # 类别、模型路径、数据路径和训练参数
├── prepare_intent_dataset.py   # ECD 文本清洗、合成酒店样本和标注集构建
├── train_intent_classifier.py  # 分类模型训练、评估和保存
├── intent_predictor.py         # 单条 / 批量推理
├── demo_intent.py              # 固定样例分类演示
├── dataset/                    # 本地原始数据，不提交 Git
├── output/                     # 已有预处理结果、权重和检查点
└── booking_extraction/         # 独立酒店检索与预订实体抽取
```

## 准备与运行

从仓库根目录安装依赖：

```powershell
python -m pip install -r requirements.txt
```

准备基础 BGE-M3 模型，并修改 `intent_config.py` 中的 `MODEL_NAME`。训练脚本开头仍使用固定的 `D:\config\...` 缓存路径，其他机器需改为可写目录。

在仓库根目录运行：

```powershell
python models/prepare_intent_dataset.py
python models/train_intent_classifier.py
python models/intent_predictor.py
python models/demo_intent.py
```

当前预处理使用合成酒店语句构建标注训练集；ECD 电商文本用于清洗和分析，不使用旧的聚类标签作为训练标签。

输出目录继续保留原名称，避免已有训练结果和权重路径失效：

- `output/data_preprocessing/cleaned_data.json`：清洗后的文本。
- `output/data_preprocessing/sampled_labels.json`：训练标注数据。
- `output/train_model/intent_model/`：模型权重、tokenizer 和意图配置。
- `output/train_model/checkpoints/`：训练检查点。

`cluster_labels.json` 等历史中间结果保留作研究参考，当前预处理脚本不会生成或读取它们。

## 集成调用

以下示例在 `models/` 目录执行：

```python
from intent_predictor import predict_intent, predict_batch

result = predict_intent("我要订一间大床房")
print(result)
results = predict_batch(["房价多少", "能便宜吗", "有停车场吗"])
```

预测返回意图、置信度及各类别概率。完整微调权重和基础模型未随 Git 仓库上传；需要训练或另行准备匹配文件。

## 评测与部署

训练输出不替代独立测试集评测。酒店领域的长尾表达、查询价格与议价的边界、低置信度处理都需要真实样本验证；当前不提供经验证的准确率和响应时间承诺。

Flask 接口位于 `../api/intent_api.py`。详细安装、配置与展示步骤见[项目 README](../README.md)。
