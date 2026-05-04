# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load sekali saat server start
bundle = joblib.load('model_nb_anxious.pkl')
model  = bundle['model']
feats  = bundle['feature_cols']
cmap   = bundle['category_map']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        jawaban = request.get_json()

        DEFAULT = cmap['No']
        encoded = []
        for col in feats:
            raw = jawaban.get(col, None)
            if raw is None or raw == 'Not interested to say':
                encoded.append(DEFAULT)
            else:
                encoded.append(cmap.get(raw, DEFAULT))

        input_arr  = np.array(encoded).reshape(1, -1)
        prediction = model.predict(input_arr)[0]
        proba      = model.predict_proba(input_arr)[0]

        # 🔥 FIX: label terbalik di model
        berisiko = (prediction == 0)  # 0 = berisiko, 1 = tidak berisiko

        return jsonify({
            'status'          : 'success',
            'result'          : 'Berisiko Anxious' if berisiko else 'Tidak Berisiko',
            'risk'            : bool(berisiko),
            'p_anxious'       : round(float(proba[0]) * 100, 1),  # index 0 = berisiko
            'p_tidak_anxious' : round(float(proba[1]) * 100, 1),  # index 1 = tidak berisiko
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'model_nb_anxious'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)