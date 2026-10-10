from intent_predictor import predict_intent

tests = ['帮我订一间房', '今晚还有房吗', '3月15号入住', '你好', '你是谁', '能便宜点吗', '有停车场吗']
for t in tests:
    r = predict_intent(t)
    intent = r['intent']
    conf = r['confidence']
    print(f"{t} -> {intent} ({conf:.2%})")
