"""酒店向量索引构建脚本

从数据集中提取唯一酒店名, 用 bge-m3 生成 embedding, 存入 FAISS 索引。

用法: python build_hotel_index.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import booking_config as config
from hotel_index import HotelIndex


def build_index_from_dataset():
    """从数据集构建索引"""
    import pandas as pd

    print(f"[BuildIndex] 读取数据集: {config.DATASET_PATH}")
    df = pd.read_excel(config.DATASET_PATH)

    if "hotel_name" not in df.columns:
        print("[BuildIndex] ERROR: 数据集缺少 hotel_name 列")
        sys.exit(1)

    hotel_names = df["hotel_name"].dropna().unique().tolist()
    print(f"[BuildIndex] 提取到 {len(hotel_names)} 家唯一酒店")

    index = HotelIndex()
    index.build(hotel_names)
    print("[BuildIndex] 完成")


if __name__ == "__main__":
    build_index_from_dataset()
