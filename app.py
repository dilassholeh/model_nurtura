from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# load model & scaler
model = joblib.load("model_kmeans.pkl")
scaler = joblib.load("scaler.pkl")

CLUSTER_BERESIKO = 0


@app.route('/predict', methods=['POST'])
def predict():
    try:
    
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "JSON tidak ditemukan"
            }), 400

        features = data.get("features")

        if features is None:
            return jsonify({
                "status": "error",
                "message": "features tidak ditemukan"
            }), 400

        # ubah ke numpy array
        input_arr = np.array(features).reshape(1, -1)

        # scaling
        input_scaled = scaler.transform(input_arr)

        # prediksi
        cluster = model.predict(input_scaled)[0]

        # mapping hasil
        result = f"Cluster {cluster}"

        return jsonify({
            "status": "success",
            "cluster": int(cluster),
            "result": result
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok"
    })


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port
    )