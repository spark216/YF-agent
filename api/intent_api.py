"""
API意图识别服务
基于 models/intent_predictor.py 构建的意图识别接口
"""
import sys
import os

# 将 models 目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from flask import Flask, request, jsonify
from intent_predictor import IntentPredictor

app = Flask(__name__)

# 模型路径
MODEL_DIR = r"models\output\train_model\intent_model"

predictor = None


def load_predictor():
    global predictor
    if predictor is None:
        predictor = IntentPredictor(model_dir=MODEL_DIR)
    return predictor


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    单条意图识别
    Request JSON: {"text": "我要订一间大床房"}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "missing 'text' field"}), 400

    text = data["text"]
    result = load_predictor().predict(text)
    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    批量意图识别
    Request JSON: {"texts": ["房价多少", "能打八折吗", "有停车场吗"]}
    """
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "missing 'texts' field"}), 400

    texts = data["texts"]
    results = load_predictor().predict_batch(texts)
    return jsonify({"results": results})


if __name__ == "__main__":
    load_predictor()
    app.run(host="0.0.0.0", port=5000, debug=True)
