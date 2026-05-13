"""
verify.py — sanity-check the from-scratch linear algebra implementations
against NumPy. Prints PASS/FAIL for each function.
Run from this folder:
    python verify.py
"""

import numpy as np
from linear_algebra import (
    dot_product,
    magnitude,
    cosine_similarity,
    mat_vec_multiply,
    project_data,
    is_orthogonal,
    forward_pass,
    attention_scores,
    transpose,
    l1_norm,
    l2_norm,
    mat_mult,
    low_rank_approximation,
)

TOL = 1e-9  # floating-point tolerance


def check(name, mine, theirs, tol=TOL):
    """Compare scalars, vectors, or matrices. Report PASS/FAIL."""
    mine_arr = np.array(mine, dtype=float)
    theirs_arr = np.array(theirs, dtype=float)
    if mine_arr.shape != theirs_arr.shape:
        print(f"FAIL  {name}: shape {mine_arr.shape} vs {theirs_arr.shape}")
        return
    if np.allclose(mine_arr, theirs_arr, atol=tol):
        print(f"PASS  {name}")
    else:
        diff = np.max(np.abs(mine_arr - theirs_arr))
        print(f"FAIL  {name}: max abs diff = {diff}")


# 1. dot_product vs np.dot
a, b = [1, 2, 3], [4, 5, 6]
check("dot_product", dot_product(a, b), np.dot(a, b))

# 2. magnitude vs np.linalg.norm
a = [1, 6, 7, 23]
check("magnitude", magnitude(a), np.linalg.norm(a))

# 3. cosine_similarity vs (np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))
a, b = [6, 3, 3], [4, 1, 9]
check("cosine_similarity", cosine_similarity(a, b), (np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))))

# 4. mat_vec_multiply vs A @ v
matrix = [
    [1, 2, 3],
    [4, 5, 6]
]
vector = [1, 0, 1]
check("mat_vec_multiply", mat_vec_multiply(matrix, vector), np.array(matrix) @ vector)

# 5. project_data vs data @ principal_component
data = [
    [5, 1, 4],  # User 1
    [4, 1, 5],  # User 2
    [2, 5, 2],  # User 3
    [1, 5, 1],  # User 4
    [3, 3, 3],  # User 5
]
principal_component = [1, -1, 1]
check("project_data", project_data(data, principal_component), np.array(data) @ principal_component)

# 6. is_orthogonal — pick a known-orthogonal pair and a non-orthogonal pair;
# compare  bool against (np.dot(a, b) == 0). Using check() with ints (0/1).
a, b = [1, 0, 0], [0, 1, 0]  # orthogonal: dot == 0
check("is_orthogonal (orthogonal)", is_orthogonal(a, b), (np.dot(a, b) == 0))
a, b = [1, 2, 3], [4, 5, 6]  # non-orthogonal: dot != 0
check("is_orthogonal (non-orthogonal)", is_orthogonal(a, b), (np.dot(a, b) == 0))

# 7. forward_pass vs chained matrix multiplications in numpy
layers = [
    [[1, 0], [0, 1]],  
    [[2, 0], [0, 2]],  
]
input_vector = [3, 4]
numpy_result = np.array(input_vector, dtype=float)
for layer in layers:
    numpy_result = np.array(layer) @ numpy_result
check("forward_pass", forward_pass(layers, input_vector), numpy_result)

# 8. attention_scores vs keys @ query
query = [1, 0, 1]
keys = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
check("attention_scores", attention_scores(query, keys), np.array(keys) @ query)

# 9. transpose vs A.T
matrix = [[1, 2, 3], [4, 5, 6]]
check("transpose", transpose(matrix), np.array(matrix).T)

# 10. l1_norm vs np.linalg.norm(a, ord=1)
a = [1, 2, 4, 7, 2, 8]
check("l1_norm", l1_norm(a), np.linalg.norm(a, ord=1))

# 11. l2_norm vs np.linalg.norm(a, ord=2)
a = [1, 2, 2, 14, 5, 6, 2]
check("l2_norm", l2_norm(a), np.linalg.norm(a, ord=2))

# 12. mat_mult vs A @ B
A = [[1, 2, 3],
     [4, 5, 6]] # 2 x 3

B = [[7,  8],
     [9,  10],
     [11, 12]] # 3 x 2
check("mat_mult", mat_mult(A, B), np.array(A) @ np.array(B))

# 13. low_rank_approximation:
#       - generate a random matrix A
#       - U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
#       - call your low_rank_approximation(U.tolist(), sigma.tolist(), Vt.tolist(), k=2)
#       - compare against the numpy reconstruction:
#           U[:, :2] @ np.diag(sigma[:2]) @ Vt[:2, :]
np.random.seed(42)
A = np.random.rand(5, 4)
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

k = 2
numpy_result = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

check("low_rank_approximation", low_rank_approximation(U.tolist(), sigma.tolist(), Vt.tolist(), k=k),numpy_result)