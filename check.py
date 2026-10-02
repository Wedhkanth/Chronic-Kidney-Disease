import pandas as pd
import joblib
import shap

dt = joblib.load("models/dt_model.pkl")
features = joblib.load("models/features.pkl")
scaler = joblib.load("models/scaler.pkl")

df = pd.DataFrame([[-0.1]*len(features)], columns=features)
scaled = scaler.transform(df)

explainer = shap.TreeExplainer(dt)
shap_values = explainer.shap_values(df)
import numpy as np

print("IS LIST?", isinstance(shap_values, list))
if isinstance(shap_values, list):
    print("Len of list:", len(shap_values))
    print("Array shape in list:", shap_values[0].shape)
else:
    print("Shape:", getattr(shap_values, 'shape', 'No shape'))
    
ev = explainer.expected_value
print("Expected value:", type(ev), getattr(ev, '__len__', 'No len'), ev)
