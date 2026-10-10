"""全局配置"""
import os

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "..")

# bge-m3 向量模型 (已有)
BGE_M3_PATH = os.path.join(MODELS_DIR, "models--BAAI--bge-m3", "snapshots", "5617a9f61b028005a4858fdac845db406aefb181")

# Qwen2.5-1.5B-Instruct 模型路径 (需下载)
QWEN_PATH = os.path.join(MODELS_DIR, "models--Qwen2.5-1.5B-Instruct")

# FAISS 索引保存路径
FAISS_INDEX_PATH = os.path.join(PROJECT_DIR, "hotel_index.faiss")
FAISS_METADATA_PATH = os.path.join(PROJECT_DIR, "hotel_index_meta.json")

# 数据集路径
DATASET_PATH = os.path.join(MODELS_DIR, "dataset", "hotel_dialog_dataset.xlsx")

# 设备自动选择
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# 检索参数
TOP_K = 3
SIMILARITY_THRESHOLD = 0.5
