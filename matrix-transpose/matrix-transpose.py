import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    r,c = A.shape
    At = np.zeros((c, r))
    for i in range(r):
        for j in range(c):
            At[j][i] = A[i][j]


    return At
            
        