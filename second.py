import pandas as pd

# 1.Create a DataFrame with Name, Age, and Salary.
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 32, 40, 29, 35],
    'Salary': [50000, 60000, 75000, 48000, 68000]}

df=pd.DataFrame(data)
print(df)

# 2. Add a Bonus column (10% of Salary)
df['Bonus']=df['Salary']*0.10

# 3. Filter Age > 30
filt=df[df['Age']>30]

# Print results
print("\n",df)
print("\nFiltered data:\n",filt)
