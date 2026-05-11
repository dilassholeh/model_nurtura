import joblib
import numpy as np

bundle = joblib.load('model_1.pkl')
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

# Cek: di data asli, berapa % orang dengan banyak jawaban Yes yang BENAR-BENAR anxious?
df_check = df[available_features + ['target']].copy()

# Hitung berapa fitur yang dijawab 'Yes' (encoded = 0) per baris
df_check['jumlah_yes'] = (df_check[available_features] == 0).sum(axis=1)

print(pd.crosstab(
    df_check['jumlah_yes'],
    df_check['target'],
    rownames=['Jumlah jawaban Yes'],
    colnames=['Target'],
    margins=True
))