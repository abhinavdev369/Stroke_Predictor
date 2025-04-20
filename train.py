# train.py
# Purpose: Train a neural network to predict stroke risk, encode categorical features,
# scale numerical features, create a SHAP explainer, and save all artifacts for use in app.py.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import joblib
import numpy as np
import shap
import logging
import os

# Setup logging to track progress and debug issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load dataset from dataset/ folder
logger.info("Loading Dataset...")
df = pd.read_csv(os.path.join("dataset", "stroke_data1.csv"))
logger.info("Dataset columns: %s", df.columns.tolist())  # Debug: verify column names

# Data cleaning: remove 'id' and fill missing BMI values with mean
df.drop('id', axis=1, inplace=True)
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Encode categorical variables using LabelEncoder
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le_dict = {}
for col in categorical_cols:
    try:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    except KeyError as e:
        logger.error("Column %s not found in dataset.", col)
        raise

# Save label encoders to model/ folder
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(le_dict, os.path.join(model_dir, "label_encoders.pkl"))
logger.info("Label encoders saved.")

# Separate features and target
X = df.drop('stroke', axis=1)
y = df['stroke']
feature_names = X.columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
logger.info("Scaler saved.")

# Build neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.3),  # Prevent overfitting
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
logger.info("Training Model...")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Save the model to model/ folder
model.save(os.path.join(model_dir, "stroke_predictor_model.h5"))
logger.info("Model saved.")

# Create SHAP explainer for model interpretability
logger.info("Creating SHAP explainer...")
background_data = X_train_scaled[:100]  # Use 100 samples for efficiency
logger.info("Background data shape: %s", background_data.shape)  # Should be (100, n_features)
explainer = shap.KernelExplainer(model.predict, background_data)
shap_path = os.path.join(model_dir, "shap_explainer.pkl")
joblib.dump(explainer, shap_path)
logger.info(f"SHAP explainer saved to {shap_path}")
joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
logger.info("Feature names saved.")

logger.info("Training completed successfully!")