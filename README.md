# House-Price-Predictor
# House Price Prediction

A machine learning project that predicts residential property prices using various house attributes such as living area, number of bedrooms, bathrooms, location, and other structural features. The project compares multiple regression algorithms, evaluates their performance using cross-validation, and selects the best-performing model for price prediction.

---

## Project Overview

The objective of this project is to build an end-to-end regression model capable of estimating house prices from historical housing data. The workflow includes data preprocessing, feature engineering, model comparison, evaluation, and prediction.

The project follows a standard machine learning pipeline and is designed to be easily extended with additional models or deployment frameworks.

---

## Features

* Data cleaning and preprocessing
* Missing value handling
* Feature engineering
* One-Hot Encoding for categorical variables
* Feature scaling using `StandardScaler`
* Comparison of multiple regression algorithms
* 5-Fold Cross Validation
* Model evaluation using multiple metrics
* Visualization of model performance
* Modular and reusable code structure

---

## Algorithms Used

* Linear Regression
* K-Nearest Neighbors Regressor
* Decision Tree Regressor
* Random Forest Regressor
* Support Vector Regressor (Linear, Polynomial, and RBF kernels)

> Additional models such as XGBoost, CatBoost, Gradient Boosting, and LightGBM can be integrated without changing the project structure.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Project Structure

```text
House_Price_Predictor/
│
├── data/
│   └── Housing.csv
│
├── src/
│   ├── Feature_Enginneering.py
│   ├── model_training.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── best_model.pkl
│
├── screenshots/
│   ├── correlation_heatmap.png
│   ├── price_distribution.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Machine Learning Pipeline

1. Load the dataset
2. Clean and preprocess the data
3. Handle missing values
4. Perform feature engineering
5. Encode categorical variables
6. Split data into training and testing sets
7. Train multiple regression models
8. Perform 5-Fold Cross Validation
9. Compare model performance
10. Select the best model
11. Evaluate on the test dataset
12. Generate predictions

---

## Evaluation Metrics

The following metrics are used to compare model performance:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Cross Validation Score

---

## Results

Each algorithm is evaluated using 5-Fold Cross Validation and tested on unseen data. The final model is selected based on overall performance rather than a single metric, providing a more reliable estimate of generalization.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

Navigate to the project directory:

```bash
cd House-Price-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the training script:

```bash
python src/model_training.py
```

Run prediction:

```bash
python src/predict.py
```

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
* Integration of XGBoost, CatBoost, and LightGBM
* Feature importance analysis
* SHAP-based model interpretability
* Streamlit web application for real-time predictions
* Model deployment using Docker and cloud platforms

---

## Author

**Krish Jha**

Electronics and Communication Engineering (ECE) Student

Interested in Machine Learning, Artificial Intelligence, Data Science, and Software Development.

---

## License
