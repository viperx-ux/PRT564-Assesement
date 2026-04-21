import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("nsw_crash_preprocessed.csv")

"""
RESEARCH QUESTION
Is there a relationship between the number of traffic units involved
in a crash and the number of casualties reported?
"""

# Select variables
df_model = df[[
    "No. of traffic units involved",
    "Total_casualties"
]].dropna()

# Separate X and y
X = df_model[["No. of traffic units involved"]]
y = df_model["Total_casualties"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0
)

# ===============================
# 1. SIMPLE LINEAR REGRESSION MODEL
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# Regression equation
feature_names = X.columns
terms = [f"({coef:.4f} * {name})" for coef, name in zip(model.coef_, feature_names)]
equation = f"Total_casualties = {model.intercept_:.4f} + " + " + ".join(terms)

print("\nRegression equation:")
print(equation)

# Evaluation
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

r2 = metrics.r2_score(y_test, y_pred)

print("\nSimple Linear Regression performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("NRMSE:", rmse_norm)
print("R^2:", r2)

# ===============================
# 2. BASELINE MODEL
# ===============================
y_base = y_train.mean()
y_pred_base = [y_base] * len(y_test)

mae_base = metrics.mean_absolute_error(y_test, y_pred_base)
mse_base = metrics.mean_squared_error(y_test, y_pred_base)
rmse_base = math.sqrt(mse_base)
rmse_norm_base = rmse_base / (y_max - y_min)
r2_base = metrics.r2_score(y_test, y_pred_base)

print("\nBaseline performance:")
print("MAE:", mae_base)
print("MSE:", mse_base)
print("RMSE:", rmse_base)
print("NRMSE:", rmse_norm_base)
print("R^2:", r2_base)

# ===============================
# 3. STATSMODELS OLS
# ===============================
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

print("\nStatsmodels OLS Summary:")
print(model_sm.summary())

# ===============================
# 4. VISUALISATION
# ===============================

# Scatter plot + regression line
plot_data = pd.DataFrame({
    "x": X_test["No. of traffic units involved"],
    "y_pred": y_pred
}).sort_values("x")

plt.figure(figsize=(6, 5))
plt.scatter(X_test["No. of traffic units involved"], y_test, alpha=0.7)
plt.plot(plot_data["x"], plot_data["y_pred"])
plt.xlabel("Number of Traffic Units")
plt.ylabel("Total Casualties")
plt.title("Simple Linear Regression")
plt.savefig("fig8_simple_regression.png", dpi=300, bbox_inches="tight")
plt.close()

# Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Total Casualties")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("fig9_residual_plot_simple.png", dpi=300, bbox_inches="tight")
plt.close()

# Actual vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Total Casualties")
plt.ylabel("Predicted Total Casualties")
plt.title("Actual vs Predicted")
plt.savefig("fig10_actual_vs_predicted_simple.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 5. INTERPRETATION
# ===============================
print("\nInterpretation:")
print("The results suggest a relationship between the number of traffic units involved")
print("and the total number of casualties.")
print("A positive coefficient means that crashes with more traffic units tend to have more casualties.")
print(f"The model explains {r2:.2%} of the variation in total casualties.")
print("Compared with the baseline model, the regression model helps explain crash outcomes better.")
