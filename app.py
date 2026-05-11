from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# load model & scaler
model = joblib.load("model_kmeans.pkl")
scaler = joblib.load("scaler.pkl")

CLUSTER_BERESIKO = 0  # hasil kamu tadi

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if features is None:
            return jsonify({
                "status": "error",
                "message": "features tidak ditemukan"
            }), 400

        # ubah ke array
        input_arr = np.array(features).reshape(1, -1)

        # scaling (WAJIB)
        input_scaled = scaler.transform(input_arr)

        # prediksi cluster
        cluster = model.predict(input_scaled)[0]

        # mapping ke hasil
        result = "Berisiko" if cluster == CLUSTER_BERESIKO else "Tidak Berisiko"

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


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="10.10.6.174", port=5000, debug=True)