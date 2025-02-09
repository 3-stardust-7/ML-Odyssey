import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Split data before scaling
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df.iloc[:, :-1], df['species'], test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Try different values of K
k_values = range(1, 16)
accuracy_scaled = []
accuracy_unscaled = []

for k in k_values:
    knn_scaled = KNeighborsClassifier(n_neighbors=k)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_scaled)
    accuracy_scaled.append(accuracy_score(y_test, y_pred_scaled))
    
    knn_unscaled = KNeighborsClassifier(n_neighbors=k)
    knn_unscaled.fit(X_train_raw, y_train)
    y_pred_unscaled = knn_unscaled.predict(X_test_raw)
    accuracy_unscaled.append(accuracy_score(y_test, y_pred_unscaled))

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_scaled, marker='o', label="Scaled Data", linestyle='dashed')
plt.plot(k_values, accuracy_unscaled, marker='s', label="Unscaled Data", linestyle='dotted')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs. K (With & Without Scaling)")
plt.legend()
plt.grid()
plt.show()











# # species.py
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn import datasets

# # Load dataset
# iris = datasets.load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['species'] = iris.target

# # Feature Scaling
# scaler = StandardScaler()
# df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['species'], test_size=0.2, random_state=42)

# # Train KNN
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# # Predictions & Accuracy
# y_pred = knn.predict(X_test)
# print("KNN Accuracy:", accuracy_score(y_test, y_pred))
