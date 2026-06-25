import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from Feature_Enginneering import df

import pickle 

features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "House_age",
    "Age_of_renovation",
    "total_area",
    "renovated",
    "city",
    "statezip"
]

X = df[features].copy()

X = X.fillna(X.median(numeric_only=True))

X = pd.get_dummies(
    X,
    columns=["city", "statezip"],
    drop_first=True
)

y = np.log1p(df["price"])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


models = {
    "Linear Regression": LinearRegression(),

    "KNN": KNeighborsRegressor(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=500,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),

    "AdaBoost":
        AdaBoostRegressor(
            random_state=42
        ),

    "SVR":
        SVR(kernel="rbf")
}


results = []

best_score = -np.inf
best_model = None
best_name = ""

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    cv_score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    ).mean()

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    results.append([
        name,
        cv_score,
        r2,
        mae,
        rmse
    ])

    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        best_name = name


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "CV R2",
        "Test R2",
        "MAE",
        "RMSE"
    ]
)

results_df = results_df.sort_values(
    by="Test R2",
    ascending=False
)

print("\n==============================")
print(results_df)
print("==============================")


print(f"\nBest Model : {best_name}")
print(f"Best Test R2 : {best_score:.4f}")


y_pred_log = best_model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

print("\nFinal Metrics (Original Price Scale)")
print("----------------------------------")
print("R2 :", r2_score(y_actual, y_pred))
print("MAE:", mean_absolute_error(y_actual, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_actual, y_pred)))


plt.figure(figsize=(8,6))

plt.scatter(
    y_actual,
    y_pred,
    alpha=0.6
)

plt.plot(
    [y_actual.min(), y_actual.max()],
    [y_actual.min(), y_actual.max()],
    color="red"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")

plt.show()


residuals = y_actual - y_pred

plt.figure(figsize=(8,6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    color="red"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()
