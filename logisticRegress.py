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