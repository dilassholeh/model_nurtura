import joblib
import numpy as np

bundle = joblib.load('model_nb_anxious.pkl')
model = bundle['model']
feats = bundle['feature_cols']
cmap  = bundle['category_map']

print("Classes:", model.classes_)

# Test semua Yes
encoded_yes = [cmap['Yes']] * 8
pred = model.predict(np.array(encoded_yes).reshape(1, -1))[0]
proba = model.predict_proba(np.array(encoded_yes).reshape(1, -1))[0]
print(f"Semua 'Yes' → prediksi: {pred}, proba: {proba}")

# Test semua No
encoded_no = [cmap['No']] * 8
pred2 = model.predict(np.array(encoded_no).reshape(1, -1))[0]
proba2 = model.predict_proba(np.array(encoded_no).reshape(1, -1))[0]
print(f"Semua 'No'  → prediksi: {pred2}, proba: {proba2}")