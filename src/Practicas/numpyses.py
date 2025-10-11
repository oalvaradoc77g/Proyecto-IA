import numpy as np

# Creating arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.zeros((3, 3))
array3 = np.ones((2, 4))
random_array = np.random.rand(3, 3)

# Basic operations
print("Basic array:", array1)
print("Sum of elements:", array1.sum())
print("Mean:", array1.mean())
print("Max value:", array1.max())

# Array operations
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

print("\nMatrix multiplication:")
print(np.dot(matrix1, matrix2))

# Array reshaping
array4 = np.array([1, 2, 3, 4, 5, 6])
reshaped = array4.reshape(2, 3)
print("\nReshaped array:")
print(reshaped)

# Array slicing
print("\nSliced array:")
print(array1[1:4])  # Get elements from index 1 to 3