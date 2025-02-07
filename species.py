# Steps:
# Basic statistics: Mean, median, correlation
# Pairplot (Seaborn): Quick visualization of relationships
# Boxplot & Violin plot: Understand distribution
# Histogram: See feature distributions
# Scatterplot: See feature relationships

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets


# Load the Iris dataset
iris=datasets.load_iris()

# Convert to DataFrame
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=iris.target
df['species']=df['species'].map({0:'setosa', 1: 'versicolor', 2: 'virginica'})

# Basic stats
print(df.describe())
# print(df.corr())
print(df.corr(numeric_only=True))


# Set Seaborn style
sns.set_style("whitegrid")

# Pairplot
sns.pairplot(df,hue="species",diag_kind="kde")
plt.show()


