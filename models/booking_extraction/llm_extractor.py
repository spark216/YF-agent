"""Qwen2.5-1.5B-Instruct 本地模型加载与推理"""
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import booking_config as config

SYSTEM_PROMPT = """你是一个酒店预订信息抽取助手。

任务: 从用户对话中抽取酒店名、入住日期、离店日期。

输出要求:
- 必须且只能输出一个合法 JSON 对象
- 不输出任何额外文字、解释或 markdown 格式
- 酒店名使用提供的标准名称
- 日期格式: YYYY-MM-DD
- 如果某个字段无法确定, 用 null 表示

输出格式:
{"hotel_name": "标准酒店名", "checkin_date": "YYYY-MM-DD", "checkout_date": "YYYY-MM-DD", "confidence": 0.95}"""


class LLMExtractor:
    """Qwen2.5-1.5B 本地推理器"""

    def __init__(self):
        self.device = config.DEVICE
        self.tokenizer = None
        self.model = None

    def load(self):
        """加载模型 (懒加载)"""
        if self.model is not None:
            return

        model_path = config.QWEN_PATH
        print(f"[LLMExtractor] 加载模型: {model_path}")
        print(f"[LLMExtractor] 设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        print("[LLMExtractor] 模型加载完成")

    def extract(self, query: str, candidate_hotels: list[tuple[str, float]]) -> dict:
        """
        调用 Qwen 抽取结构化信息

        Args:
            query: 用户原始对话
            candidate_hotels: [(酒店名, 相似度), ...]

        Returns:
            {"hotel_name": str, "checkin_date": str, "checkout_date": str, "confidence": float}
        """
        self.load()

        hotel_list_str = "\n".join([f"- {name} (相似度: {score:.4f})" for name, score in candidate_hotels])

        user_content = f"""请从以下对话中抽取酒店名和日期信息。

标准酒店列表:
{hotel_list_str}

用户对话:
{query}

请严格输出 JSON:"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        result_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return self._parse_json(result_text, candidate_hotels)

    def _parse_json(self, text: str, candidates: list[tuple[str, float]]) -> dict:
        """从 LLM 输出中提取 JSON"""
        # 尝试提取 JSON 块
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # 兜底: 返回最高相似度候选酒店 + null 日期
        if candidates:
            best_name, best_score = candidates[0]
            return {
                "hotel_name": best_name,
                "checkin_date": None,
                "checkout_date": None,
                "confidence": round(best_score, 4),
            }
        return {
            "hotel_name": None,
            "checkin_date": None,
            "checkout_date": None,
            "confidence": 0.0,
        }


_extractor: LLMExtractor | None = None


def get_extractor() -> LLMExtractor:
    global _extractor
    if _extractor is None:
        _extractor = LLMExtractor()
    return _extractor
