from datetime import datetime, timedelta, timezone

from flask import Flask, request, jsonify
from google_play_scraper import Sort, reviews
from google_play_scraper.constants.element import ElementSpec, ElementSpecs
from transformers import pipeline

app = Flask(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Keep the raw [seconds, nanoseconds] pair: the library's default converts to
# local time and drops the sub-second part.
ElementSpecs.Review["at"] = ElementSpec(None, [5])

print("Loading DistilBERT model...")
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
print("Model ready.")


def to_utc_iso(raw):
    seconds = raw[0]
    nanos = raw[1] if len(raw) > 1 and raw[1] else 0
    moment = datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(milliseconds=nanos // 1_000_000)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@app.route("/reviews", methods=["GET"])
def latest_reviews():
    result, _ = reviews(
        request.args.get("app_id", "com.ubercab"),
        lang=request.args.get("lang", "en"),
        country=request.args.get("country", "us"),
        sort=Sort.NEWEST,
        count=int(request.args.get("count", 200)),
    )
    return jsonify([
        {
            "reviewId": r["reviewId"],
            "content": r["content"],
            "reviewCreatedVersion": r["reviewCreatedVersion"] if r["reviewCreatedVersion"] is not None else "Unknown",
            "at": to_utc_iso(r["at"]),
            "thumbsUpCount": r["thumbsUpCount"] if r["thumbsUpCount"] is not None else 0,
        }
        for r in result
    ])


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