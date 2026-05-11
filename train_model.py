import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("Loading data...")
df = pd.read_csv('predictor/data/accidents.csv')

# Features and target
X = df.drop('severity', axis=1)
y = df['severity']

# Identify categorical and numerical columns
categorical_cols = ['weather', 'road_type']
numerical_cols = ['lat', 'long', 'hour', 'speed_limit', 'visibility']

print("Building pipeline...")
# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and the classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train / Test split to check accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model_pipeline.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the entire pipeline (includes encoders and the trained model)
os.makedirs('predictor/ml_models', exist_ok=True)
joblib.dump(model_pipeline, 'predictor/ml_models/accident_model.pkl')
print("Model saved to predictor/ml_models/accident_model.pkl")