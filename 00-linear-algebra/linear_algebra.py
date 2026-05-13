
import math
from typing import Sequence

# Write a Python function from scratch — no NumPy — that takes two vectors as lists and returns their cosine similarity.

def dot_product(a: Sequence[float], b: Sequence[float]) -> float:

    if len(a) != len(b):
        raise ValueError("Vectors must be the same length")
    return sum(x * y for x,y in zip(a, b))

def magnitude(a: Sequence[float]) -> float:
    return math.sqrt(dot_product(a, a))

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    # compute cosine similarity of a to b: (a dot b)/{||a||*||b||)
    return dot_product(a, b) / (magnitude(a) * magnitude(b) )

# Write a matrix-vector multiplication from scratch — no NumPy:
def mat_vec_multiply(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    # matrix is a list of rows
    # each row dotted with the vector gives one output element
    # Multiplies a matrix by a vector from scratch, returning the resulting vector.
    # The number of columns in the matrix must equal the length of the vector.
    # Get the number of rows (M) and columns (N) from the matrix shape
    if len(matrix) == 0:
        return []
    if len(matrix[0]) != len(vector):
        raise ValueError(f"Matrix columns ({len(matrix[0])}) must match vector length ({len(vector)})")
    
    return [dot_product(row, vector) for row in matrix]
           
# Given a dataset of vectors and one principal component vector, project every data point onto that component.
def project_data(data: Sequence[Sequence[float]], principal_component: Sequence[float]) -> list[float]:
    # project each data point onto the principal component
    # hint: projection of a point onto a direction = dot_product(point, principal_component)
    # return a list of scalar values — one per data point
    return [dot_product(row, principal_component) for row in data]
    
# Write a function that hecks if two vectors are orthogonal.
def is_orthogonal(a: Sequence[float], b: Sequence[float]) -> bool:
    
    return dot_product(a, b) == 0


# Write a function that runs a forward pass through a neural network represented as a list of matrices.
def forward_pass(layers: Sequence[Sequence[Sequence[float]]], input_vector: Sequence[float]) -> list[float]:
    # layers is a list of matrices
    # apply each matrix to the vector in sequence
    new_vector: list[float] = list(input_vector)
    for layer in layers:

        new_vector = mat_vec_multiply(layer, new_vector)
    
    return new_vector

# Write a function that takes a query vector and a list of key vectors, and returns a relevance score for each key.
def attention_scores(query: Sequence[float], keys: Sequence[Sequence[float]]) -> list[float]:
    # compute dot product between query and each key
    # return a list of scores
    return [dot_product(query, key) for key in keys]

def transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    # flip rows and columns
    # hint: element [i][j] becomes element [j][i]
    return [list(row) for row in zip(*matrix)]

def l1_norm(a: Sequence[float]) -> float:
    # sum of absolute values
    return sum(abs(elements) for elements in a)

def l2_norm(a: Sequence[float]) -> float:
    #square -> sum -> square root
    return math.sqrt(sum(x**2 for x in a))

def low_rank_approximation(U, sigma, Vt, k):
    # U is a matrix (list of lists)
    # sigma is a list of singular values
    # Vt is a matrix (list of lists)
    # k is how many singular values to keep

    # 1. keep only first k columns of U
    U_k = [row[:k] for row in U]
    # 2. keep only first k singular values
    sigma_k = sigma[:k]
    # 3. keep only first k rows of Vt
    Vt_k = Vt[:k]
    # 4. multiply them back together: U_k × diag(sigma_k) × Vt_k
    # U_k * sigma_k * Vt_k
    US_k = [[u * s for u, s in zip(row, sigma_k)] for row in U_k]

    return mat_mult(US_k, Vt_k)

    
def mat_mult(A, B):
    B_t = transpose(B) # columns of B become rows, now iterable as vectors
    result_cols = [mat_vec_multiply(A, col) for col in B_t]
    return  transpose(result_cols)


