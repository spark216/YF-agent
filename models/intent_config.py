"""
酒店客服用户意图识别 - 配置模块
"""
import os

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 意图分类定义
INTENT_NAMES = ["查询价格", "下单预订", "议价砍价", "常规咨询"]
INTENT_TO_IDX = {name: idx for idx, name in enumerate(INTENT_NAMES)}
IDX_TO_INTENT = {idx: name for idx, name in enumerate(INTENT_NAMES)}
NUM_INTENTS = len(INTENT_NAMES)

# 数据路径
ECD_DATA_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "DeepUtteranceAggregation-master",
    "DeepUtteranceAggregation-master",
    "ECD_sample",
)

# 处理后的数据路径 (prepare_intent_dataset.py 输出)
BASE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
DATA_PREPROC_DIR = os.path.join(BASE_OUTPUT_DIR, "data_preprocessing")
CLEANED_DATA_PATH = os.path.join(DATA_PREPROC_DIR, "cleaned_data.json")
EMBEDDINGS_PATH = os.path.join(DATA_PREPROC_DIR, "embeddings.npz")
CLUSTER_LABELS_PATH = os.path.join(DATA_PREPROC_DIR, "cluster_labels.json")
SAMPLED_LABELS_PATH = os.path.join(DATA_PREPROC_DIR, "sampled_labels.json")

# BGE-M3 模型配置
MODEL_NAME = os.path.join(PROJECT_DIR, "models--BAAI--bge-m3", "snapshots", "5617a9f61b028005a4858fdac845db406aefb181")
MAX_LENGTH = 128
EMBEDDING_DIM = 1024

# HDBSCAN 聚类参数
UMAP_DIM = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_MIN_CLUSTER_SIZE = 15

# 采样参数 (每类意图)
SAMPLE_SIZE_PER_INTENT = 40  # 4类 × 40 = 160条

# 微调训练参数
TRAIN_EPOCHS = 6
TRAIN_LR = 2e-5
TRAIN_BATCH_SIZE = 16
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_WARMUP_RATIO = 0.1
TRAIN_MAX_GRAD_NORM = 1.0

# 模型保存路径 (train_intent_classifier.py 输出)
TRAIN_MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, "train_model")
MODEL_SAVE_DIR = os.path.join(TRAIN_MODEL_DIR, "intent_model")
CHECKPOINT_DIR = os.path.join(TRAIN_MODEL_DIR, "checkpoints")
