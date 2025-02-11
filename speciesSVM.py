import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]  # Selecting only two features for visualization (sepal length, petal length)
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Train SVM models with different kernels
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

svm_poly = SVC(kernel='poly', C=1.0, degree=3)
svm_poly.fit(X_train, y_train)

# Plot decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# KNN Decision Boundary
plot_decision_regions(X_train, y_train, clf=knn, ax=axes[0])
axes[0].set_title("KNN Decision Boundary")

# SVM Linear Decision Boundary
plot_decision_regions(X_train, y_train, clf=svm_linear, ax=axes[1])
axes[1].set_title("SVM Linear Decision Boundary")

# SVM RBF Decision Boundary
plot_decision_regions(X_train, y_train, clf=svm_rbf, ax=axes[2])
axes[2].set_title("SVM RBF Decision Boundary")

plt.show()
