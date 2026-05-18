from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId

app = Flask(__name__)

# load model & scaler
model = joblib.load("model_kmeans.pkl")
scaler = joblib.load("scaler.pkl")

CLUSTER_BERESIKO = 0

# Mapping for answers to numerical values
mapping = {
    'Not at all': 0,
    'No': 0,
    'Maybe': 1,
    'Not interested to say': 1,
    'Sometimes': 2,
    'Often': 3,
    'Yes': 4,
    'Two or more days a week': 4
}

# MongoDB connection
mongo_uri = os.environ.get("MONGODB_URI", "mongodb+srv://userNurtura:nurturame123@cluster0.2vph2f5.mongodb.net/DBnurtura?authSource=admin")
client = MongoClient(mongo_uri)
db = client['DBnurtura']
health_records = db['health_records']
prediction_results = db['prediction_results']
users_collection = db['users']


@app.route('/predict', methods=['POST'])
def predict():
    try:
    
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "JSON tidak ditemukan"
            }), 400

        answers = data.get("answers")
        mother_id = data.get("mother_id")

        if answers is None or mother_id is None:
            return jsonify({
                "status": "error",
                "message": "answers atau mother_id tidak ditemukan"
            }), 400
        
        # Validasi mother_id ada di collection users
        try:
            print(f"DEBUG: Checking mother_id: {mother_id}")
            mother_obj_id = ObjectId(mother_id)
            print(f"DEBUG: Created ObjectId: {mother_obj_id}")
            
            mother = users_collection.find_one({"_id": mother_obj_id})
            print(f"DEBUG: Found mother: {mother is not None}")
            
            if not mother:
                print(f"DEBUG: Mother not found, returning 404")
                return jsonify({
                    "status": "error",
                    "message": "Mother tidak ditemukan dalam sistem"
                }), 404
            
            # Validasi mother memiliki role 'mother'
            mother_role = mother.get('role')
            print(f"DEBUG: Mother role: {mother_role}")
            if mother_role != 'mother':
                return jsonify({
                    "status": "error",
                    "message": "User bukan mother"
                }), 403
        except Exception as e:
            print(f"DEBUG: Exception during validation: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Invalid mother_id: {str(e)}"
            }), 400

        # Map answers to features array based on ml_index
        features = [0] * 9  # assuming 9 features
        field_to_index = {
            "perasaan_sedih_atau_mudah_menangis": 0,
            "mudah_marah_terhadap_bayi_dan_pasangan": 1,
            "kesulitan_tidur_di_malam_hari": 2,
            "kesulitan_konsentrasi_atau_mengambil_keputusan": 3,
            "makan_berlebihan_atau_kehilangan_nafsu_makan": 4,
            "merasa_cemas": 5,
            "perasaan_bersalah": 6,
            "kesulitan_membangun_ikatan_dengan_bayi": 7,
            "percobaan_bunuh_diri": 8
        }

        for field, answer in answers.items():
            if field in field_to_index:
                features[field_to_index[field]] = mapping.get(answer, 0)

        # ubah ke numpy array
        input_arr = np.array(features).reshape(1, -1)

        # scaling
        input_scaled = scaler.transform(input_arr)

        # prediksi
        cluster = model.predict(input_scaled)[0]

        # mapping hasil
        result = "Beresiko Depresi" if cluster == CLUSTER_BERESIKO else "Tidak Beresiko Depresi"

        # Simpan ke health_records
        health_record = {
            "mother_id": ObjectId(mother_id),
            "created_at": datetime.utcnow(),
            **{k: str(v) for k, v in answers.items()}  # convert to string
        }
        health_insert = health_records.insert_one(health_record)
        health_id = health_insert.inserted_id

        # Simpan ke prediction_results
        prediction_result = {
            "mother_id": ObjectId(mother_id),
            "result": result,
            "created_at": datetime.utcnow(),
            "health_record_id": health_id
        }
        prediction_results.insert_one(prediction_result)

        return jsonify({
            "status": "success",
            "cluster": int(cluster),
            "result": result
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

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