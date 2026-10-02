from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

app = Flask(__name__)

# Load models
lgb = joblib.load("models/lgb_model.pkl")
cb = joblib.load("models/cb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

# Load feature selection rankings (if available)
rankings = None
if os.path.exists("models/feature_rankings.pkl"):
    rankings = joblib.load("models/feature_rankings.pkl")

# ==============================
# Feature display names for the form
# ==============================
FEATURE_LABELS = {
    'age': 'Age',
    'bp': 'Blood Pressure',
    'sg': 'Specific Gravity',
    'al': 'Albumin',
    'su': 'Sugar',
    'bgr': 'Blood Glucose Random',
    'bu': 'Blood Urea',
    'sc': 'Serum Creatinine',
    'sod': 'Sodium',
    'pot': 'Potassium',
    'hemo': 'Hemoglobin',
    'pcv': 'Packed Cell Volume',
    'wc': 'White Blood Cell Count',
    'rc': 'Red Blood Cell Count',
    'rbc': 'Red Blood Cells',
    'pc': 'Pus Cell',
    'pcc': 'Pus Cell Clumps',
    'ba': 'Bacteria',
    'htn': 'Hypertension',
    'dm': 'Diabetes Mellitus',
    'cad': 'Coronary Artery Disease',
    'appet': 'Appetite',
    'pe': 'Pedal Edema',
    'ane': 'Anemia',
    'egfr': 'eGFR (auto-calculated)',
}

# eGFR
def calculate_eGFR(age, sc):
    return 175 * (sc ** -1.154) * (age ** -0.203)


# ==============================
# Home
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# About
# ==============================
@app.route("/about")
def about():
    return render_template("about.html")


# ==============================
# Feature Selection Results
# ==============================
@app.route("/feature-selection")
def feature_selection():
    has_chart = os.path.exists("static/feature_selection_comparison.png")
    return render_template(
        "feature_selection.html",
        rankings=rankings,
        features=features,
        has_chart=has_chart
    )


# ==============================
# Predict
# ==============================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Build dynamic input fields (exclude 'egfr' — it's auto-calculated)
    input_fields = [f for f in features if f != 'egfr']
    if 'egfr' in features:
        if 'age' not in input_fields:
            input_fields.append('age')
        if 'sc' not in input_fields:
            input_fields.append('sc')
    
    field_labels = {f: FEATURE_LABELS.get(f, f.upper()) for f in input_fields}


    if request.method == "POST":
        try:
            data = {}

            for f in input_fields:
                val = request.form.get(f)
                data[f] = float(val) if val not in ["", None] else 0

            # eGFR calculation (if egfr is in features and we have age + sc)
            if 'egfr' in features and 'age' in data and 'sc' in data:
                data['egfr'] = calculate_eGFR(data['age'], data['sc'])

            # Convert to DataFrame
            df = pd.DataFrame([data])

            # Feature alignment
            df = df.reindex(columns=features, fill_value=0)

            # Scaling
            scaled = scaler.transform(df)

            # Prediction
            lgb_pred = lgb.predict(scaled)[0]
            cb_pred = cb.predict(scaled)[0]

            final = int((lgb_pred + cb_pred) >= 1)

            result = "CKD Detected" if final else "No CKD Detected"

            # eGFR value for display
            egfr_val = data.get('egfr', None)
            if egfr_val:
                egfr_val = round(egfr_val, 2)

            # ==============================
            # SHAP (Patient-level Explanation)
            # ==============================
            shap_img = None
            try:
                explainer = shap.TreeExplainer(lgb)

                shap_values = explainer.shap_values(df)
                
                sv = np.array(shap_values)
                ev = explainer.expected_value
                
                # If shape is (n_samples, n_features, n_classes), extract positive class
                if len(sv.shape) == 3:
                    sv = sv[..., 1]
                
                # Fallback if list
                if isinstance(shap_values, list):
                    sv = shap_values[1]
                
                # Make it 1D for a single-prediction matplotlib plot
                if len(sv.shape) == 2 and sv.shape[0] == 1:
                    sv = sv[0]
                    
                # Handle expected_value array
                if hasattr(ev, '__len__') and len(ev) > 1:
                    ev = ev[1]

                shap.force_plot(
                    ev,
                    sv,
                    df.iloc[0, :],
                    matplotlib=True,
                    show=False
                )

                plt.savefig("static/user_shap.png", bbox_inches='tight', dpi=150)
                plt.close()

                shap_img = "static/user_shap.png"

            except Exception as e:
                print("SHAP error:", e)

            return render_template(
                "predict.html",
                prediction=result,
                egfr=egfr_val,
                shap_img=shap_img,
                input_fields=input_fields,
                field_labels=field_labels,
                features=features
            )

        except Exception as e:
            print("ERROR:", e)
            return render_template(
                "predict.html",
                error=str(e),
                input_fields=input_fields,
                field_labels=field_labels,
                features=features
            )

    return render_template(
        "predict.html",
        input_fields=input_fields,
        field_labels=field_labels,
        features=features
    )


if __name__ == "__main__":
    app.run(debug=True)