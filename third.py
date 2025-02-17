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


# LINE PLOT
# A line plot displays data points on a 2D graph connected by a straight line.
# It's typically used to visualize trends over a continuous variable, such as time or distance.
# x: A list of x-values (horizontal axis).
# y: A list of y-values (vertical axis).
# marker='o': Adds circle markers at each data point.
# linestyle='-': Connects the data points with a solid line.
# color='b': Sets the line color to blue (b stands for blue).
# xlabel() and ylabel() add labels to the x-axis and y-axis respectively.
# title() adds a title to the plot.
# grid(True) enables the grid lines, which makes it easier to read the plot.
# show() displays the plot.

# BAR CHART
# categories: A list of categorical labels ('A', 'B', 'C', etc.).
# values: A list of numerical values corresponding to each category.
# bar() creates a bar chart with categories on the x-axis and values on the y-axis.
# color='green': Sets the bar color to green
# xlabel() and ylabel() add labels to the x-axis and y-axis respectively.
# title() adds a title to the bar chart.
# show() displays the chart.
