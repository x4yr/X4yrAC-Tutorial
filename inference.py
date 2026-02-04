"""
Run MLSAC model as HTTP API. For local test or deploy (e.g. HF Spaces).
POST / with body: {"inputs": {"data": "<base64 float32 bytes>"}}
Returns: [{"probability": float}]
"""

import base64
import json
import os
import sys

# Add parent so pipeline can be imported when run from model/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import MLSACPipeline

pipeline = MLSACPipeline()


def handler(raw_input):
    if isinstance(raw_input, dict) and "inputs" in raw_input:
        inputs = raw_input["inputs"]
    else:
        inputs = raw_input
    return pipeline(inputs)


if __name__ == "__main__":
    try:
        from flask import Flask, request, jsonify
        app = Flask(__name__)

        @app.route("/", methods=["GET"])
        def index():
            return "MLSAC Inference API. POST / or POST /predict with body: {\"inputs\": {\"data\": \"<base64>\"}}", 200

        @app.route("/", methods=["POST"])
        @app.route("/predict", methods=["POST"])
        def predict():
            try:
                body = request.get_json() or {}
                out = handler(body)
                prob = out[0].get("probability", 0) if out else 0
                print(f"POST / predict -> probability={prob:.4f}", flush=True)
                return jsonify(out)
            except Exception as e:
                print(f"POST / error: {e}", flush=True)
                return jsonify({"error": str(e)}), 400

        port = int(os.environ.get("PORT", 7860))
        app.run(host="0.0.0.0", port=port)
    except ImportError:
        # No Flask: read JSON from stdin, print result
        data = json.load(sys.stdin)
        result = handler(data)
        print(json.dumps(result))
