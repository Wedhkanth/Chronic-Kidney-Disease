import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

lgb = joblib.load("models/lgb_model.pkl")
features = joblib.load("models/features.pkl")
scaler = joblib.load("models/scaler.pkl")

# dummy data
df = pd.DataFrame([[0]*len(features)], columns=features)
scaled = scaler.transform(df)

try:
    explainer = shap.TreeExplainer(lgb)
    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        ev = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) or (hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > 1) else explainer.expected_value
        sv = shap_values[1]
    else:
        ev = explainer.expected_value
        sv = shap_values

    print("Type of sv:", type(sv))
    if hasattr(sv, 'shape'):
        print("Shape of sv:", sv.shape)
    print("Type of ev:", type(ev))
    print("Type of df:", type(df))

    shap.force_plot(
        ev,
        sv,
        df,
        matplotlib=True,
        show=False
    )
    print("Success without index!")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    if hasattr(sv, 'shape') and len(sv.shape) == 2:
        sv_1d = sv[0]
    else:
        sv_1d = sv
    shap.force_plot(ev, sv_1d, df.iloc[0, :], matplotlib=True, show=False)
    print("Success WITH index!")
except Exception as e:
    print("Second try failed too:", e)

