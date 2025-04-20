# app.py
# Purpose: Serve a Flask web application that allows users to input patient data via a form,
# predict stroke risk using the trained model, and display results with SHAP explanations
# and personalized recommendations.

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange
import os
from flask_wtf.csrf import CSRFProtect
import shap

# Initialize Flask app with CSRF protection
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Secure key for forms
csrf = CSRFProtect(app)

# Load model and preprocessors from model/ folder
model_dir = "model"
try:
    model = load_model(os.path.join(model_dir, "stroke_predictor_model.h5"))
    le_dict = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    shap_path = os.path.join(model_dir, "shap_explainer.pkl")
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"SHAP explainer file not found at {shap_path}.")
    explainer = joblib.load(shap_path)
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Function to generate personalized lifestyle recommendations
def get_recommendations(features, shap_values):
    recs = []
    # Compare numerical features with scaled thresholds
    if float(features[0][7]) > scaler.mean_[7] + scaler.scale_[7]:  # High glucose
        recs.append("Reduce sugar intake by 10% daily.")
    if float(features[0][9]) == le_dict['smoking_status'].transform(['smokes'])[0]:  # Smokes
        recs.append("Consider quitting smoking to lower risk.")
    if float(features[0][8]) > scaler.mean_[8] + scaler.scale_[8]:  # High BMI
        recs.append("Aim for a 5% weight reduction through exercise.")
    return recs

# Define the input form using Flask-WTF
class PredictionForm(FlaskForm):
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    age = FloatField('Age', validators=[DataRequired(), NumberRange(min=0, max=120)])
    hypertension = SelectField('Hypertension', choices=[(0, 'No'), (1, 'Yes')], validators=[DataRequired()])
    heart_disease = SelectField('Heart Disease', choices=[(0, 'No'), (1, 'Yes')], validators=[DataRequired()])
    ever_married = SelectField('Ever Married', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    work_type = SelectField('Work Type', choices=[('Private', 'Private'), ('Self-employed', 'Self-employed'), 
                                                  ('Govt_job', 'Government'), ('children', 'Children'), 
                                                  ('Never_worked', 'Never Worked')], validators=[DataRequired()])
    residence_type = SelectField('Residence Type', choices=[('Urban', 'Urban'), ('Rural', 'Rural')], validators=[DataRequired()])
    avg_glucose_level = FloatField('Avg Glucose Level', validators=[DataRequired(), NumberRange(min=0)])
    bmi = FloatField('BMI', validators=[DataRequired(), NumberRange(min=0)])
    smoking_status = SelectField('Smoking Status', choices=[('formerly smoked', 'Formerly Smoked'), 
                                                            ('never smoked', 'Never Smoked'), 
                                                            ('smokes', 'Smokes'), ('Unknown', 'Unknown')], 
                                 validators=[DataRequired()])
    submit = SubmitField('Predict')

# Route for the main page and prediction
@app.route('/', methods=['GET', 'POST'])
def home():
    form = PredictionForm()
    if form.validate_on_submit():
        try:
            # Encode categorical inputs
            categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            encoded_vals = []
            for col in categorical_cols:
                form_key = 'residence_type' if col == 'Residence_type' else col
                try:
                    encoded_val = le_dict[col].transform([form.data[form_key]])[0]
                    encoded_vals.append(encoded_val)
                except KeyError as e:
                    return render_template('index.html', form=form, prediction_text=f"Error: Encoder not found for {col}")
                except ValueError as e:
                    return render_template('index.html', form=form, prediction_text=f"Error: Invalid value for {col}: {form.data[form_key]}")

            # Create feature array with explicit type conversion
            features = np.array([[encoded_vals[0], float(form.age.data), float(form.hypertension.data), float(form.heart_disease.data),
                                  encoded_vals[1], encoded_vals[2], encoded_vals[3], float(form.avg_glucose_level.data),
                                  float(form.bmi.data), encoded_vals[4]]], dtype=float)

            # Scale features and predict
            features_scaled = scaler.transform(features)
            prediction_prob = model.predict(features_scaled)[0][0]

            # Compute SHAP values for interpretability
            shap_values = explainer.shap_values(features_scaled)
            risk_factors = {feature_names[i]: float(shap_values[0][i]) for i in range(len(feature_names))}
            top_factors = sorted(risk_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

            # Generate recommendations
            recommendations = get_recommendations(features, shap_values)

            # Prepare results
            result = 'Stroke Risk' if prediction_prob > 0.5 else 'No Stroke Risk'
            risk_score = prediction_prob * 100

            return render_template('index.html', form=form, prediction_text=result, risk_score=risk_score,
                                   top_factors=top_factors, recommendations=recommendations)
        except Exception as e:
            return render_template('index.html', form=form, prediction_text=f"Error: {str(e)}")
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)