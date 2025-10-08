import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('data/car_data.csv')

# Select features and target
features = ['year', 'make', 'model', 'mileage', 'engine_size', 'fuel_type', 'transmission']
target = 'price'

X = df[features]
y = df[target]

# Preprocessing for categorical features
categorical_features = ['make', 'model', 'fuel_type', 'transmission']
numeric_features = ['year', 'mileage', 'engine_size']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Try different ML algorithms
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse:.2f}")

# Example: Predict price for a new car
def predict_price(car_details, pipeline):
    df_input = pd.DataFrame([car_details])
    return pipeline.predict(df_input)[0]

# Example usage
car_details = {
    'year': 2018,
    'make': 'Toyota',
    'model': 'Corolla',
    'mileage': 35000,
    'engine_size': 1.8,
    'fuel_type': 'Petrol',
    'transmission': 'Automatic'
}

# Use the best model (e.g., Random Forest)
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
best_pipeline.fit(X, y)
predicted_price = predict_price(car_details, best_pipeline)
print(f"Predicted price: ${predicted_price:.2f}")