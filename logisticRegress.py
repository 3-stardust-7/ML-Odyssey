import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data[:, :2]  # Taking only two features for visualization
y = (iris.target != 0).astype(int)  # Converting to binary (Class 0 vs. Others)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#visualisation
# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 100),
                     np.linspace(X[:,1].min(), X[:,1].max(), 100))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
probs = model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.3)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={0: "blue", 1: "red"})
plt.title("Logistic Regression Decision Boundary")
plt.show()





"""

numpy as np: Used for numerical computations and handling arrays.
matplotlib.pyplot as plt: Used for data visualization (plotting graphs).
pandas as pd: Though imported, it's not used in the script. Normally used for handling tabular data.
seaborn as sns: A data visualization library built on top of Matplotlib.
train_test_split: Used to split data into training and testing sets.
StandardScaler: Used for feature scaling (standardization).
LogisticRegression: A machine learning model for binary classification.
accuracy_score, classification_report, confusion_matrix: Metrics to evaluate the model's performance.
load_iris: Loads the Iris dataset.

iris.data[:, :2]: Selects only the first two features (columns) from the dataset for easier visualization.
iris.target: Contains labels (0, 1, or 2).
(iris.target != 0).astype(int): Converts it into a binary classification problem:
If target == 0 → label remains 0
If target != 0 (i.e., 1 or 2) → label becomes 1.

train_test_split(X, y, test_size=0.2, random_state=42): Splits X and y into training (80%) and testing (20%) datasets.
random_state=42 ensures reproducibility.

Creates a mesh grid covering the range of feature values.
np.linspace(X[:,0].min(), X[:,0].max(), 100): Generates 100 equally spaced points along the first feature axis.
np.linspace(X[:,1].min(), X[:,1].max(), 100): Generates 100 points along the second feature axis.
np.meshgrid(): Creates a 2D grid from these points.

xx.ravel() and yy.ravel() flatten the meshgrid into a list of points.
np.c_[]: Combines them into a 2-column array (matching feature format).
scaler.transform(grid): Scales the grid using the same transformation as training data.
model.predict_proba(grid_scaled)[:, 1]: Gets probability predictions for class 1.
.reshape(xx.shape): Reshapes back into the meshgrid's shape.

sns.scatterplot(): Plots the data points.
hue=y: Colors points by class (0: blue, 1: red).
plt.title(): Sets the title.
plt.show(): Displays the plot.

"""