# Car Price Prediction

This project predicts the price of a used car based on its features using machine learning. It includes data analysis, model training (with multiple algorithms), and a web app for interactive price prediction.

## Features

- Data cleaning and preprocessing
- Model training with XGBoost, Random Forest, and Linear Regression
- Label encoding for categorical features
- Model evaluation (RMSE, R²)
- Streamlit web app for user-friendly predictions
- Example notebooks for analysis and experimentation

## Project Structure

```
car_price_prediction/
│
├── car_price_app.py           # Streamlit web app for predictions
├── car_price_prediction.py    # ML training and evaluation script
├── carprice.ipynb             # Jupyter notebook for EDA and modeling
├── carPricepredict.ipynb      # Additional notebook (experiments)
├── car_price_xgb_model.pkl    # Trained XGBoost model
├── label_encoders.pkl         # Saved label encoders for categorical features
├── processed_car_dataset.csv  # Cleaned and processed dataset
├── requirements.txt           # Python dependencies
├── data/
│   └── dataset.csv            # Raw car dataset
└── README.md                  # Project documentation
```

## Getting Started

### 1. Install Dependencies & Run Project

Create a virtual environment and install requirements:

```sh
python -m venv env
.\env\Scripts\activate  # (check for folder)
pip install -r requirements.txt
streamlit run car_price_app.py
```

### 2. Prepare Data

- Place your raw dataset as `data/dataset.csv`.
- Run the notebook [`carprice.ipynb`](carprice.ipynb) to clean data, encode features, and train models.
- This will generate `processed_car_dataset.csv`, `car_price_xgb_model.pkl`, and `label_encoders.pkl`.

### 3. Train & Evaluate Models

- Use [`car_price_prediction.py`](car_price_prediction.py) to experiment with different ML algorithms (Random Forest, Linear Regression, etc.).
- Evaluate model performance using RMSE and R² metrics.

### 4. Run the Web App

Start the Streamlit app to predict car prices interactively:

```sh
streamlit run car_price_app.py
```

Enter car details in the UI to get an instant price estimate.

## Example Usage

- Try the sample prediction in [`carprice.ipynb`](carprice.ipynb) or use the web app.
- The app supports features like brand, model, year, mileage, engine size, fuel type, transmission, condition, color, body type, etc.

## Requirements

- Python 3.8+
- See [`requirements.txt`](requirements.txt) for all dependencies.

## Credits

- Built with [scikit-learn](https://scikit-learn.org/), [xgboost](https://xgboost.readthedocs.io/), [pandas](https://pandas.pydata.org/), and [Streamlit](https://streamlit.io/).

---

**Note:** For best results, use a dataset with similar columns as in `data/dataset.csv`. You can further tune models or add new
