import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle
import json

# 1. Load & Clean
df = pd.read_csv('Autism.csv')
df.replace('?', np.nan, inplace=True)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[df['age'] <= 100].dropna(subset=['age'])

# 2. Preprocessing
data = df.drop(['age_desc', 'used_app_before'], axis=1)

# We need to save the encoders to use them in the Streamlit app
encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

X = data.drop(['Class/ASD', 'result'], axis=1)
y = data['Class/ASD']

# 3. Train Production Model
model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=5, 
    eval_metric='logloss',
    random_state=42
)
model.fit(X, y)

# 4. Save Assets
# Save as Booster (bypasses sklearn requirement for inference)
model.get_booster().save_model('autism_model.bin')

# Save encoders and feature names for the app
metadata = {
    'features': list(X.columns),
    'target_names': ['Negative', 'Positive'],
    'classes': {col: list(le.classes_) for col, le in encoders.items() if col != 'Class/ASD'}
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f)

print("Production Assets Exported: autism_model.bin/json, model_metadata.json")
