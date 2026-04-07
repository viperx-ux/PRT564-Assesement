import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------
# Load preprocessed dataset
# -------------------------------
df = pd.read_csv("nsw_crash_preprocessed.csv")

# -------------------------------
# Multiple Linear Regression
# -------------------------------
FEATURES = [
    "No. of traffic units involved",
    "Speed_limit_num",
    "Is_dark",
    "Is_wet",
    "Is_weekend"
]

TARGET = "Total_casualties"

# -------------------------------
# Prepare data
# -------------------------------
data = df[FEATURES + [TARGET]].dropna()

X = data[FEATURES]
y = data[TARGET]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# Evaluation metrics
# -------------------------------
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("REGRESSION RESULTS")
print(f"R2   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# -------------------------------
# Coefficients
# -------------------------------
coef_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": model.coef_
})

print("\nCoefficients:")
print(coef_df)

# -------------------------------
# Regression equation
# -------------------------------
equation = f"{model.intercept_:.4f} + " + " + ".join(
    [f"({coef:.4f} * {feat})" for coef, feat in zip(model.coef_, FEATURES)]
)

print("\nRegression Equation:")
print(f"{TARGET} = {equation}")

# -------------------------------
# OLS summary and p-values
# -------------------------------
X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()

print("\nOLS Summary:")
print(ols_model.summary())

print("\nP-values:")
print(ols_model.pvalues)

# -------------------------------
# Figure 8: Actual vs Predicted
# -------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Total Casualties")
plt.ylabel("Predicted Total Casualties")
plt.title("Figure 8: Actual vs Predicted")
plt.savefig("fig8_Actual_vs_Predicted.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------
# Figure 9: Residual Plot
# -------------------------------
residuals = y_test - y_pred

plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Total Casualties")
plt.ylabel("Residuals")
plt.title("Figure 9: Residual Plot")
plt.savefig("fig9_Residual_Plot.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------
# Interpretation
# -------------------------------
print("\nInterpretation:")
print(f"The model explains {r2:.2%} of the variation in total casualties.")
print("Traffic and environmental variables show measurable relationships with crash outcomes.")
print("The p-values from OLS help identify which predictors are statistically significant.")
