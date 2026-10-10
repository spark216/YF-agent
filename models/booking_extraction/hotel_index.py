"""FAISS 酒店向量索引 + RAG 检索"""
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import booking_config as config


class HotelIndex:
    """FAISS 酒店向量库, 支持模糊匹配、简称、错别字容忍"""

    def __init__(self):
        self.index_path = config.FAISS_INDEX_PATH
        self.meta_path = config.FAISS_METADATA_PATH
        self.model = None
        self.index = None
        self.hotel_names = []

    def _load_encoder(self):
        if self.model is None:
            self.model = SentenceTransformer(config.BGE_M3_PATH, device=config.DEVICE)
            print(f"[HotelIndex] bge-m3 已加载: {config.BGE_M3_PATH}")

    def build(self, hotel_names: list[str]):
        """从酒店名列表构建 FAISS 索引"""
        self._load_encoder()
        self.hotel_names = list(dict.fromkeys(hotel_names))
        embeddings = self.model.encode(self.hotel_names, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"hotel_names": self.hotel_names}, f, ensure_ascii=False)
        print(f"[HotelIndex] 索引已构建: {len(self.hotel_names)} 家酒店, dim={dim}")

    def load(self) -> bool:
        """加载已有索引"""
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            print("[HotelIndex] 索引文件不存在, 请先运行 build_hotel_index.py")
            return False
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.hotel_names = meta["hotel_names"]
        print(f"[HotelIndex] 已加载: {len(self.hotel_names)} 家酒店")
        return True

    def search(self, query: str, top_k: int = config.TOP_K) -> list[tuple[str, float]]:
        """检索最相似的酒店名, 返回 [(name, score), ...]"""
        self._load_encoder()
        if self.index is None:
            self.load()
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((self.hotel_names[idx], float(score)))
        return results


_hotel_index: HotelIndex | None = None


def get_hotel_index() -> HotelIndex:
    global _hotel_index
    if _hotel_index is None:
        _hotel_index = HotelIndex()
        if not _hotel_index.load():
            _hotel_index.build(_sample_hotels())
    return _hotel_index


def _sample_hotels() -> list[str]:
    """从数据集提取酒店列表 (fallback)"""
    try:
        import pandas as pd
        df = pd.read_excel(config.DATASET_PATH)
        if "hotel_name" in df.columns:
            return df["hotel_name"].dropna().unique().tolist()
    except Exception:
        pass
    return []
