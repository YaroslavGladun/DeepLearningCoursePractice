import numpy as np
from tester import *


#
# Task 1. Create array. It must be one line of code!!!
#

# 1.1. Create array with shape (n, m) filled with zeros.
def solve_1_1(n: int, m: int) -> np.ndarray:
    return np.zeros((n, m))


assert test_solution_1_1(solve_1_1)


# 1.2. Write a Python function that takes two natural numbers a and b as input, where a < b.
# The function should return an array X of dimension (100,) such that X[1] - X[0] = X[2] - X[1] = ... = X[99] - X[98].
# The first element of the array X should be 'a' and the last element should be 'b'.
def solve_1_2(a: float, b: float) -> np.ndarray:
    return np.linspace(a, b, 100)


assert test_solution_1_2(solve_1_2)


# 1.3. The function solve_1_3 takes an integer n as input
# and returns an n-dimensional array A that satisfies the following conditions:
# - The first column of A (A[0, 0], A[1, 0], ..., A[n-1, 0]) is filled with zeros.
# - The last row of A (A[n-1, 0], A[n-1, 1], ..., A[n-1, n-1]) is filled with zeros.
# - The submatrix of A excluding the last row and the first column (A[:-1, 1:]) forms an identity matrix.
def solve_1_3(n: int) -> np.ndarray:
    return np.eye(n, k=1)


assert test_solution_1_3(solve_1_3)


#
# Task 2. Operations
#

# 2.1. Argmax

# 2.2. Implemet to gray

# 2.3.

#
# Task 3. Indexes and slices
#

# 3.1. You have RGB image with shape (n, m, 3). Convert it to BGR.
def solve_3_1(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1]


assert test_solution_3_1(solve_3_1)


# 3.2. You have RGB image with shape (1398, 2145, 3). Implement vertical flip for this image.
def solve_3_2(image: np.ndarray) -> np.ndarray:
    return image[:, ::-1, :]


assert test_solution_3_2(solve_3_2)


# 3.3. You have RGB image with shape (1398, 2145, 3). Implement rescale for this image to shape (1600, 1600, 3).
def solve_3_3(image: np.ndarray) -> np.ndarray:
    row_indexes = np.linspace(0, image.shape[0] - 1, 1600).astype(np.int32)
    col_indexes = np.linspace(0, image.shape[1] - 1, 1600).astype(np.int32)
    return image[row_indexes.reshape(-1, 1), col_indexes]


assert test_solution_3_3(solve_3_3)

#
# Task 5. https://algotester.com/en/ArchiveProblem/DisplayWithEditor/40456
#
