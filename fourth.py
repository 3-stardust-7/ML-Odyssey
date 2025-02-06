# 4️⃣ scikit-learn Exercise
# Train a simple Linear Regression model.

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 1. Define data
x=np.array([1,2,3,4,5]).reshape(-1,1) # Reshape is needed
y=np.array([2,3,5,7,11])

# 2. Train model
model=LinearRegression()
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
