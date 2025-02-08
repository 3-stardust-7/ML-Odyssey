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

# OTHER VISUALISATIONS
#Boxplot (Feature Distribution by Species)
# plt.figure(figsize=(10, 6))
# sns.boxplot(x="species", y="sepal length (cm)", data=df)
# plt.show()

# #Violin Plot (Density)
# plt.figure(figsize=(10, 6))
# sns.violinplot(x="species", y="petal length (cm)", data=df)
# plt.show()

# #Histogram (Feature Distribution)
# df.hist(figsize=(10, 6), bins=20)
# plt.show()

# #Scatterplot (Sepal Length vs. Petal Length)
# sns.scatterplot(x=df["sepal length (cm)"], y=df["petal length (cm)"], hue=df["species"])
# plt.show()

