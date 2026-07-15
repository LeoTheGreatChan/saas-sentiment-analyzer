from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print("Loading DistilBERT model...")
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
print("Model ready.")


@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    result = sentiment_pipeline(str(text)[:512])[0]
    score = result["score"] if result["label"] == "POSITIVE" else -result["score"]

    return jsonify({
        "label": result["label"].capitalize(),
        "score": round(score, 4),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)