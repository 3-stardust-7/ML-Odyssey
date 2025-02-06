# Create a simple line plot and bar chart.

import matplotlib.pyplot as plt


# 1. Line plot
x=[1,2,3,4,5]
y=[2,3,5,7,11]

# lots of plt func there
plt.plot(x,y,marker='o',linestyle='-',color='b')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot Example')
plt.grid(True)
plt.show()

# 2. Bar chart
categories=['A','B','C','D','E']
values=[10,20,30,15,30]
plt.bar(categories,values,color='green')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()