"""
脚本3: 意图识别推理服务

提供两种接口:
  1. 实时单条推理: predict_intent(text) -> {"intent": str, "prob": float, "probs": dict}
  2. 批量预测: predict_batch(texts) -> List[dict]

特点:
  - 自动适配CPU/GPU环境
  - 单次推理 <10ms (GPU) / <50ms (CPU)
  - 模型即插即用，无需外部服务依赖
"""
import os
import json
import time
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput

import intent_config as config
from train_intent_classifier import IntentClassifier


def _resolve_model_path(model_name: str) -> str:
    """将相对路径解析为绝对路径（基于 models/ 目录）"""
    if os.path.isabs(model_name):
        return model_name
    return os.path.join(config.PROJECT_DIR, model_name)


# ============================================================
# 意图识别推理器
# ============================================================

class IntentPredictor:
    """
    酒店客服意图识别推理器

    支持:
      - 单条文本实时识别 (<50ms)
      - 批量文本离线识别
      - CPU/GPU自动切换

    Usage:
        predictor = IntentPredictor(model_dir="output/intent_model")
        result = predictor.predict("我要订一间大床房")
        print(result)
        # {'intent': '下单预订', 'confidence': 0.95, 'all_probs': {...}}

        results = predictor.predict_batch(["房价多少", "能便宜吗", "有停车场吗"])
    """

    def __init__(self, model_dir=config.MODEL_SAVE_DIR):
        """
        加载微调后的模型

        Args:
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载意图配置
        intent_config_path = os.path.join(model_dir, "intent_config.json")
        with open(intent_config_path, "r", encoding="utf-8") as f:
            self.intent_config = json.load(f)

        self.intent_names = self.intent_config["intent_names"]
        self.idx_to_intent = {int(k): v for k, v in self.intent_config["idx_to_intent"].items()}
        self.max_length = self.intent_config["max_length"]

        # 加载Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True,
        )

        # 加载模型
        model_name = _resolve_model_path(self.intent_config["model_name"])
        self.model = IntentClassifier(
            model_name=model_name,
            num_labels=self.intent_config["num_intents"],
        )
        self.model.load_state_dict(
            torch.load(
                os.path.join(model_dir, "pytorch_model.bin"),
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载完成: {model_dir}")
        print(f"  设备: {self.device}")
        print(f"  意图类别: {self.intent_names}")

    # ----------------------------------------------------------
    # 单条实时推理
    # ----------------------------------------------------------

    def predict(self, text: str) -> Dict:
        """
        实时单条推理

        Args:
            text: 用户输入文本

        Returns:
            {
                "intent": str,       # 意图名称
                "confidence": float, # 最高概率
                "all_probs": dict,   # 所有意图的概率
            }
        """
        # 预处理
        text = self._preprocess(text)
        if not text:
            return self._empty_result()

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 推理
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()
            all_probs = probs[0].tolist()

        return {
            "intent": self.idx_to_intent[pred_idx],
            "confidence": round(confidence, 4),
            "all_probs": {
                self.intent_names[i]: round(p, 4)
                for i, p in enumerate(all_probs)
            },
        }

    def predict_with_timing(self, text: str) -> Dict:
        """带耗时的推理 (性能测试用)"""
        start = time.perf_counter()
        result = self.predict(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result["elapsed_ms"] = round(elapsed_ms, 2)
        return result

    # ----------------------------------------------------------
    # 批量离线预测
    # ----------------------------------------------------------

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[Dict]:
        """
        批量预测

        Args:
            texts: 文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度

        Returns:
            List[Dict]: 每条文本的预测结果
        """
        results = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._predict_batch_internal(batch)
            results.extend(batch_results)

            if show_progress and (i + batch_size) % (batch_size * 10) == 0:
                print(f"  已处理: {min(i + batch_size, total)}/{total}")

        return results

    def _predict_batch_internal(self, texts: List[str]) -> List[Dict]:
        """内部批量推理 (单批次)"""
        # 预处理
        cleaned = [self._preprocess(t) if t else "" for t in texts]

        # 过滤空文本
        valid_indices = [i for i, t in enumerate(cleaned) if t]
        valid_texts = [cleaned[i] for i in valid_indices]

        # 初始化结果 (空文本给默认值)
        batch_results = [self._empty_result() for _ in texts]

        if valid_texts:
            # Tokenize (批量)
            encodings = self.tokenizer(
                valid_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # 推理
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            # 组装结果
            for j, orig_idx in enumerate(range(len(valid_texts))):
                idx = valid_indices[j] if j < len(valid_indices) else j
                if j < len(valid_indices):
                    real_idx = valid_indices[j]
                    batch_results[real_idx] = {
                        "intent": self.idx_to_intent[preds[j].item()],
                        "confidence": round(probs[j][preds[j]].item(), 4),
                        "all_probs": {
                            self.intent_names[k]: round(probs[j][k].item(), 4)
                            for k in range(len(self.intent_names))
                        },
                    }

        return batch_results

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    @staticmethod
    def _preprocess(text: str) -> str:
        """文本预处理"""
        text = text.strip()
        text = "".join(text.split())  # 去除所有空白符
        return text

    @staticmethod
    def _empty_result() -> Dict:
        """空文本默认结果"""
        return {
            "intent": "常规咨询",
            "confidence": 0.0,
            "all_probs": {name: 0.0 for name in config.INTENT_NAMES},
        }

    # ----------------------------------------------------------
    # 性能基准测试
    # ----------------------------------------------------------

    def benchmark(self, n_samples: int = 100) -> Dict:
        """
        性能基准测试

        Args:
            n_samples: 测试样本数

        Returns:
            {
                "avg_ms": float,
                "min_ms": float,
                "max_ms": float,
                "throughput": int,  # 条/秒
            }
        """
        test_texts = [
            "房价多少", "我要订房", "能打几折", "有停车场吗",
            "最便宜的房间", "帮我预订", "便宜点吧", "WiFi密码",
            "含早价格", "订两间房", "会员折扣", "离机场多远",
        ]

        times = []
        for i in range(n_samples):
            text = test_texts[i % len(test_texts)]
            start = time.perf_counter()
            self.predict(text)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        times = np.array(times)
        return {
            "avg_ms": round(float(np.mean(times)), 2),
            "min_ms": round(float(np.min(times)), 2),
            "max_ms": round(float(np.max(times)), 2),
            "p50_ms": round(float(np.percentile(times, 50)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "p99_ms": round(float(np.percentile(times, 99)), 2),
            "throughput": round(1000 / np.mean(times)),
        }


# ============================================================
# 便捷函数
# ============================================================

# 全局单例 (避免重复加载模型)
_predictor: Optional[IntentPredictor] = None


def get_predictor(model_dir: str = config.MODEL_SAVE_DIR) -> IntentPredictor:
    """获取单例预测器 (避免重复加载)"""
    global _predictor
    if _predictor is None or _predictor.model_dir != model_dir:
        _predictor = IntentPredictor(model_dir)
    return _predictor


def predict_intent(text: str) -> Dict:
    """
    实时单条意图识别 (便捷函数)

    Args:
        text: 用户输入文本

    Returns:
        {"intent": str, "confidence": float, "all_probs": dict}

    Example:
        >>> predict_intent("我要订一间大床房")
        {'intent': '下单预订', 'confidence': 0.85, 'all_probs': {...}}
    """
    return get_predictor().predict(text)


def predict_batch(texts: List[str], batch_size: int = 32) -> List[Dict]:
    """
    批量意图识别 (便捷函数)

    Args:
        texts: 文本列表
        batch_size: 批次大小

    Returns:
        List[Dict]: 每条文本的预测结果

    Example:
        >>> predict_batch(["房价多少", "能便宜吗"])
        [{'intent': '查询价格', ...}, {'intent': '议价砍价', ...}]
    """
    return get_predictor().predict_batch(texts, batch_size=batch_size)


# ============================================================
# CLI: 演示与测试
# ============================================================

def demo():
    """交互式演示"""
    print("=" * 60)
    print("  酒店客服意图识别 - 实时推理演示")
    print("=" * 60)

    predictor = IntentPredictor()

    # --- 1. 单条推理测试 ---
    print("\n--- 单条推理测试 ---\n")
    test_cases = [
        "一间大床房多少钱",
        "帮我预订明天晚上的房间",
        "能打八折吗",
        "你们酒店有停车场吗",
    ]

    for text in test_cases:
        result = predictor.predict_with_timing(text)
        print(f"  输入: {text}")
        print(f"  意图: {result['intent']} (置信度: {result['confidence']:.2%})")
        print(f"  耗时: {result['elapsed_ms']:.1f}ms")
        print()

    # --- 2. 批量推理测试 ---
    print("--- 批量推理测试 ---\n")
    batch_texts = [
        "最便宜的房间多少钱", "我要订两间房",
        "便宜点我就住", "WiFi密码多少",
        "海景房一晚多少钱", "预订要付定金吗",
        "会员价能打几折", "早餐几点开始",
    ]

    batch_results = predictor.predict_batch(batch_texts, show_progress=True)
    print()
    for text, result in zip(batch_texts, batch_results):
        print(f"  {text:<20s} -> {result['intent']:<6s} ({result['confidence']:.2%})")

    # --- 3. 性能基准测试 ---
    print("\n--- 性能基准测试 ---\n")
    bench = predictor.benchmark(n_samples=100)
    print(f"  平均延迟: {bench['avg_ms']:.1f}ms")
    print(f"  P50延迟:  {bench['p50_ms']:.1f}ms")
    print(f"  P95延迟:  {bench['p95_ms']:.1f}ms")
    print(f"  P99延迟:  {bench['p99_ms']:.1f}ms")
    print(f"  最大延迟: {bench['max_ms']:.1f}ms")
    print(f"  吞吐量:    {bench['throughput']} 条/秒")

    # --- 4. 交互式查询 ---
    print("\n--- 交互式查询 (输入 'quit' 退出) ---\n")
    while True:
        text = input("请输入文本 > ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        result = predictor.predict_with_timing(text)
        print(f"  意图: {result['intent']} | 置信度: {result['confidence']:.2%} | 耗时: {result['elapsed_ms']:.1f}ms")
        print(f"  详细概率: {result['all_probs']}")
        print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo()
