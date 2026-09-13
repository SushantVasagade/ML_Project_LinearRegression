# Linear Regression for House Price Prediction

## Project overview

This experiment predicts house prices using demographic and local-area features. It introduces regression modelling, coefficient interpretation, residual analysis, cross-validation, and common regression error metrics.

## Problem statement

Given area income, house age, rooms, bedrooms, and population, estimate the continuous `Price` of a house. This is a **supervised regression** problem because the target is a numerical value rather than a category.

## Dataset requirements

Download or copy `USA_Housing.csv` into the same folder as the notebook. The expected columns are:

| Column | Used in model? | Description |
|---|:---:|---|
| `Avg. Area Income` | Yes | Average income of the area |
| `Avg. Area House Age` | Yes | Average age of houses in the area |
| `Avg. Area Number of Rooms` | Yes | Average rooms per house |
| `Avg. Area Number of Bedrooms` | Yes | Average bedrooms per house |
| `Area Population` | Yes | Population of the area |
| `Price` | Target | House price to predict |
| `Address` | No | Text address; excluded from basic numeric regression |

## Methodology

1. Load the CSV with pandas and verify its path.
2. Inspect data types, descriptive statistics, missing values, and duplicate records.
3. Visualize feature relationships with pair plots, price distribution, and a correlation heatmap.
4. Use five numeric input columns as `X` and `Price` as `y`.
5. Split data into 80% training and 20% testing sets using `random_state=42`.
6. Fit a scikit-learn `LinearRegression` model on training data.
7. Predict unseen test prices.
8. Evaluate MAE, MSE, RMSE, and R².
9. Plot actual versus predicted prices and residual diagnostics.
10. Inspect coefficients, run five-fold cross-validation, and save predictions.

## Evaluation metrics

| Metric | Interpretation |
|---|---|
| MAE | Average absolute difference between actual and predicted price |
| MSE | Average squared error; penalizes large errors more strongly |
| RMSE | Square root of MSE; reported in price units |
| R² | Fraction of target variation explained by the model |

## Interpreting the plots

- **Actual vs predicted plot:** points near the diagonal line indicate accurate predictions.
- **Residual histogram:** residuals centered around zero are desirable.
- **Residual vs predicted plot:** residuals should not show strong curves or changing spread.
- **Coefficient chart:** shows the expected change in price for a one-unit feature increase, holding other variables fixed.

## Files and outputs

```text
ML_LinearRegression/
├── README.md
├── Linear_Regression.ipynb
├── USA_Housing.csv
└── house_price_predictions.csv
```

The prediction CSV contains actual test prices, predicted prices, and residual values.

## Run instructions

```bash
conda activate ml
jupyter lab
```

Open `Linear_Regression.ipynb`. Confirm that the file is exactly named `USA_Housing.csv` before running the data-loading cell.

## Limitations and improvements

Linear Regression assumes a roughly linear relationship between predictors and price. It does not use the `Address` text field, interaction effects, or non-linear patterns. Possible improvements include one-hot encoding location, adding polynomial features, handling outliers, and comparing Ridge, Lasso, Random Forest, or Gradient Boosting regression models.
