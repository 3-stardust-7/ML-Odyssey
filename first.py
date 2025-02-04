import numpy as np

# 1. Create a 1D array
arr=np.array([10,20,30,40,50])

# 2. Multiply by 2
arr_2=arr*2

# 3. Compute mean and sum
mean = np.mean(arr)
sum = np.sum(arr)

# 4. Create a 3x3 random matrix (values between 1 and 10)
matrix = np.random.randint(1,11,size=(3,5))

# Print results
print("Array:",arr)
print("Array * 2:",arr_2)
print("Mean:",mean)
print("Sum:",sum)
print("Random matrix:\n",matrix)








# matrix = np.random.randint(1, 11, size=(3, 3))
# This line creates a 3x3 matrix filled with random integer0s between 1 and 10 (inclusive of 1, exclusive of 11) using NumPy.
# Syntax below
# np.random.randint(low, high, size, dtype)
# low	The minimum value (inclusive).
# high	The maximum value (exclusive).
# size	Shape of the output array (e.g., (3, 3) for a 3x3 matrix).
# dtype	The data type (default is int).
