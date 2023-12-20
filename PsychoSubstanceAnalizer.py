from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from io import BytesIO
from rdkit.Chem import Draw

# CHEMBLからIC50データを取得
def fetch_ic50_data(chembl_ids):
    activity = new_client.activity
    records = []
    for cid in chembl_ids:
        for rec in activity.filter(target_chembl_id=cid, standard_type="IC50"):
            if rec['standard_value'] is not None:
                ic50 = float(rec['standard_value'])
                pic50 = -np.log10(ic50 * 1e-9)  # IC50をpIC50に変換
                pic50_value = round(pic50, 3) if pic50 >= -3 else np.nan
                records.append({'chembl_id': cid, 'smiles': rec['canonical_smiles'], 'pic50': pic50_value})
    return pd.DataFrame(records)

# 化学特徴量の抽出
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp)

# 機械学習モデルの構築と訓練
def build_and_train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2048,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, callbacks=[early_stopping])
    return model, X_test, y_test

# GUI関連の関数
def display_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(300, 200))
    bio = BytesIO()
    img.save(bio, 'PNG')
    img = Image.open(bio)
    return ImageTk.PhotoImage(img)

def predict_affinity(model, r2, smiles):
    if smiles:
        features = extract_features(smiles)
        prediction = model.predict(np.array([features]))[0][0]
        img = display_structure(smiles)
        canvas.create_image(150, 100, image=img)
        canvas.image = img
        result_label.config(text=f'Prediction: {prediction}, R^2 Score: {r2}')
    else:
        messagebox.showerror("Error", "Invalid SMILES")

# メイン関数
def main():
    chembl_ids = [
        'CHEMBL2363064', 'CHEMBL2095205', 'CHEMBL1893', 'CHEMBL2331074',
        'CHEMBL2094109', 'CHEMBL2094268', 'CHEMBL240', 'CHEMBL2095181',
        'CHEMBL3885538', 'CHEMBL3883321', 'CHEMBL224', 'CHEMBL2111468',
        'CHEMBL2096981', 'CHEMBL3301387', 'CHEMBL2094124', 'CHEMBL2109244'
    ]

    ic50_data = fetch_ic50_data(chembl_ids)
    mean_pic50 = ic50_data['pic50'].mean()
    ic50_data['pic50'].fillna(mean_pic50, inplace=True)

    features = np.array([extract_features(sm) for sm in ic50_data['smiles'] if sm is not None])
    labels = ic50_data['pic50'].values

    model, X_test, y_test = build_and_train_model(features, labels)
    predicted = model.predict(X_test)
    r2 = r2_score(y_test, predicted)

    # GUIのセットアップ
    window = tk.Tk()
    window.title("Ligand Affinity Predictor")

    global canvas, result_label
    canvas = tk.Canvas(window, width=300, height=200)
    canvas.pack()

    iupac_entry = tk.Entry(window, width=50)
    iupac_entry.pack()

    predict_button = tk.Button(window, text="Predict Affinity", command=lambda: predict_affinity(model, r2, iupac_entry.get()))
    predict_button.pack()

    result_label = tk.Label(window, text="")
    result_label.pack()

    window.mainloop()

if __name__ == "__main__":
    main()
