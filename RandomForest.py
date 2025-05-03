from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 5. Plot feature importances
importances = clf.feature_importances_
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


"""
🌟 Key Concepts of Random Forest
Decision Trees:

A decision tree is a simple model that makes decisions based on features, splitting the data at each node according to some criterion (e.g., Gini impurity or entropy for classification).

Limitations: They tend to overfit the data, which means they can perform well on the training data but not generalize well to new data.

Ensemble Learning:

Ensemble methods combine multiple individual models to make more accurate and stable predictions.

Random Forest is an example of bagging (Bootstrap Aggregating) — it builds multiple models in parallel using different subsets of the data.


How to Tune Random Forest
Random Forest has a few key hyperparameters that you can tune for better performance:

n_estimators: The number of trees in the forest (default is 100). More trees usually lead to better performance but increase computation time.

max_depth: The maximum depth of each tree. Deeper trees can overfit, so controlling this can help with generalization.

min_samples_split: The minimum number of samples required to split an internal node. Higher values prevent the model from learning overly specific patterns.

min_samples_leaf: The minimum number of samples required to be at a leaf node. Similar to min_samples_split, this helps with generalization.

max_features: The number of features to consider when looking for the best split. Lower values make the model more random.

Visualizing Random Forest
One interesting feature of Random Forest is feature importance.
You can visualize which features contribute most to the decision-making process of the model.
This helps you understand your data better and, in some cases, identify unnecessary or redundant features.
importances = clf.feature_importances_
sns.barplot(x=importances, y=feature_names)

"""