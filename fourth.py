# 4️⃣ scikit-learn Exercise
# Train a simple Linear Regression model.

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 1. Define data
x=np.array([1,2,3,4,5]).reshape(-1,1) # Reshape is needed
y=np.array([2,3,5,7,11])

# 2. Train model
model=LinearRegression() #Creates an instance of the LinearRegression model, which can be trained on the provided data.
model.fit(x,y)

# 3. Predict for X = 6
new=np.array([[6]])
prediction=model.predict(new)

print("Predictioon for 6:",prediction[0])
# for cleaner statement,This will round the output to two decimal places (e.g., 12.20).
# print(f"Prediction for 6: {prediction[0]:.2f}")


# Scatter plot of data
plt.scatter(x, y, color='blue', label='Data points')

# Regression line
x_line = np.linspace(1, 6, 100).reshape(-1, 1)  # Generate points for smooth line
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', label='Regression Line')

# Highlighting the prediction for X=6
plt.scatter([6], prediction, color='green', marker='o', s=100, label='Prediction for X=6')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

"""
{But why reshape?
scikit-learn expects the input features (X) to be in a 2D array with the shape (n_samples, n_features).
n_samples: number of data points (rows).
n_features: number of features per data point (columns).
We have 5 data points (1, 2, 3, 4, 5).
Each data point has only 1 feature (just the number itself).

The -1 is a placeholder that tells NumPy:
"Figure out this dimension for me based on the other dimension."
Since we specified 1 for the number of columns, NumPy calculates how many rows are needed.
Here, it automatically determines there are 5 rows because there are 5 elements in total.
Example:
If we had 10 elements and did .reshape(-1, 2), it would reshape into (5, 2) because:
10 elements total ÷ 2 columns = 5 rows.

Trains (fits) the linear regression model on the data (x, y).
The model learns the best-fit line by calculating the slope and intercept that minimize the mean squared error.

new = np.array([[6]])
What it does:
Creates a 2D array [[6]] to predict the output value for x = 6.
Must be 2D because scikit-learn expects the input in that format.

Uses the trained model to predict the output for x = 6.

plt.scatter(x, y, color='blue', label='Data points')
What it does:
Plots the original data points as a scatter plot in blue.
The label parameter will be used in the legend.

x_line = np.linspace(1, 6, 100).reshape(-1, 1)  # Generate points for smooth line
What it does:
np.linspace(1, 6, 100) generates 100 evenly spaced points between 1 and 6.
.reshape(-1, 1) reshapes it into a 2D array for prediction.

y_line = model.predict(x_line)
What it does:
Predicts the output values for all the points in x_line to create a smooth regression line.

lt.scatter([6], prediction, color='green', marker='o', s=100, label='Prediction for X=6')
What it does:
Plots the predicted point (6, prediction) in green with a larger marker (s=100).
plt.xlabel('X-axis')
Adds label to the x-axis.
same for y axis}
"""