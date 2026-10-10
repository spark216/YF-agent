# 酒店检索与预订信息抽取

独立研究模块：使用 BGE-M3 + FAISS 检索酒店名，结合中文日期规则和本地 Qwen 模型，提取入住 / 离店日期。当前尚未接入闲鱼主客服图。

## 文件用途

```text
booking_extraction/
├── booking_config.py     # 模型、数据、索引路径和检索参数
├── booking_extractor.py  # extract_booking_info 主流程与演示样例
├── build_hotel_index.py  # 从酒店数据集构建向量索引
├── hotel_index.py        # 索引读写与相似酒店检索
├── date_parser.py        # 中文与相对日期解析
├── llm_extractor.py      # Qwen 加载与辅助抽取
├── hotel_index.faiss     # 已有向量索引
└── hotel_index_meta.json # 索引对应的酒店名称
```

## 准备与运行

在仓库根目录安装 `requirements.txt`。检查 `booking_config.py` 中的 BGE-M3、Qwen、酒店数据集和索引路径；索引文件不能替代模型权重。

```powershell
Set-Location "models/booking_extraction"
python build_hotel_index.py
python booking_extractor.py
```

构建索引需要 `models/dataset/hotel_dialog_dataset.xlsx`，其中包含 `hotel_name` 列。基础模型及本地 Qwen 权重需自行准备。

## 模块调用

在本目录执行：

```python
from booking_extractor import extract_booking_info

result = extract_booking_info("帮我订汉庭酒店明天入住后天离店")
print(result)
```

返回字段为 `hotel_name`、`checkin_date`、`checkout_date`、`confidence`，另含内部计时字段 `_elapsed_ms`。未检索到候选酒店时，返回空酒店和日期及零置信度。

## 流程与边界

1. 检索候选酒店，当前最终酒店名使用第一条候选。
2. 优先通过规则提取日期。
3. 日期不完整时调用本地 Qwen 补充。
4. 合并信息并返回结果。

当前置信度包含启发式调整，不等价于经过校准的正确概率；同名酒店、日期歧义和无匹配样本仍需人工或业务规则处理。性能应在实际硬件与独立样本上测量。

完整工程配置见[项目 README](../../README.md)。
