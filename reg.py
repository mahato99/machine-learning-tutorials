import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the data from CSV
data = pd.read_csv('regression_data.csv')

# Assuming the CSV has columns 'x' and 'y'
#X = data['x'].values
X = data['x'].values.reshape(-1, 1)
y = data['y'].values

# Plotting
plt.figure(figsize=(12, 6))

# Original data and regression line
plt.scatter(X, y, alpha=0.7, label='Original Data Plot')
plt.title('Linear Regression Sample Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the model parameters
estimated_slope = model.coef_[0]
estimated_intercept = model.intercept_

# Make predictions
y_pred = model.predict(X)

# Calculate performance metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Plotting
plt.figure(figsize=(12, 6))

# Original data and regression line
plt.scatter(X, y, alpha=0.7, label='Original Data')
plt.plot(X, y_pred, color='green', label='Regression Line')

plt.title('Linear Regression Analysis')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Add text box with model details
details = (f'Estimated Slope: {estimated_slope:.4f}\n'
           f'Estimated Intercept: {estimated_intercept:.4f}\n'
           f'Mean Squared Error: {mse:.4f}\n'
           f'R² Score: {r2:.4f}')
plt.annotate(details, xy=(0.05, 0.95), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
             ha='left', va='top', fontsize=10)

plt.tight_layout()
plt.show()

# Print out the results
print("Model Parameters:")
print(f"Estimated Slope: {estimated_slope:.4f}")
print(f"Estimated Intercept: {estimated_intercept:.4f}")
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Prediction function
def predict_new_x(new_x):
   
    return model.predict(np.array(new_x).reshape(-1, 1))
newx=20
newy=predict_new_x(newx)[0]
print(f"The y value for X = {newx}  is : {newy}")