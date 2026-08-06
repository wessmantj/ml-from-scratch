# Linear Algebra work pytest suite
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
import pytest 

# --- dot_product tests ---

@pytest.mark.parametrize("a,b", [
    ([1, 2, 3], [4, 5, 6]),
    ([1.5, -2.0], [0.0, 4.0]),   # floats + negatives
    ([5], [3]),                   # length-1
    ([0, 0, 0], [1, 2, 3]),       # zero vector
])
def test_dot_product_correctness(a, b):
    # Checks a normal vector pair computes to the same output as numpy function
    assert dot_product(a, b) == pytest.approx(np.dot(a, b))

def test_dot_product_empty_vectors():
    # Checks edge casing empty vectors
    assert dot_product([], []) == 0

def test_dot_product_length_mismatch():
    # Tests for ValueError where lengths do not match
    with pytest.raises(ValueError):
        dot_product([1, 2], [1, 2, 3])

# -- magnitude test ---

@pytest.mark.parametrize("a", [
    [1, 6, 7, 23],
    [0.0, -3.5],
    [5],
])
def test_magnitude_correctness(a):
    # Checks a normal vector computes to the same output as numpy function
    assert magnitude(a) == pytest.approx(np.linalg.norm(a))

# --- cosine_similarity test ---

@pytest.mark.parametrize("a,b", [
    ([6, 3, 3], [4, 1, 9]),
    ([2, 1, 3], [3, 2, -1]),
    ([3, 6, 7], [7, 7, 7])
])
def test_cosine_similarity_correctness(a, b):
    # Checks a normal vector pair computes the same output as numpy function
    assert cosine_similarity(a, b) == pytest.approx(np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b))))

# --- mat_vec_multiply tests ---

@pytest.mark.parametrize("m,v", [
    ([[1, 2, 3], [4, 5, 6]], [1, 0, 1]),
    ([[3, 2, 1], [3, 7, 8]], [7, 0, 2]),
    ([[2, 4, 1], [6, 7, 7]], [-1, 7, 6])
])
def test_mat_vec_multiply_correctness(m, v):
    # Checks a normal matrix vector pair computes the same output as numpy multiplication
    assert np.allclose(mat_vec_multiply(m, v), np.array(m) @ v)

def test_mat_vec_multiply_empty_matrix():
    # Checks edge case where matrix is empty
    assert mat_vec_multiply([], [1, 2, 3]) == []

def test_mat_vec_multiply_length_mismatch():
    # Checks that length of both matrix and vector match
    with pytest.raises(ValueError):
            mat_vec_multiply([[1, 2], [2, 3]], [1, 2, 3])

# --- project_data test --- 

@pytest.mark.parametrize("d,pc", [
    ([[5, 1, 4], [4, 1, 5], [2, 5, 2], [1, 5, 1], [3, 3, 3]], [1, -1, 1]),
    ([[1.5, -2.0], [0.0, 4.0]], [2.0, 0.5]),   # floats + negatives
    ([[1, 2, 3]], [1, 0, 1]),                  # single data point
    ([[0, 0, 0], [1, 2, 3]], [4, 5, 6]),       # zero row
])
def test_project_data_correctness(d, pc):
    # Checks each data point projected onto the principal component matches numpy
    assert np.allclose(project_data(d, pc), np.array(d) @ pc)

# --- is_orthogonal test ---

def test_is_orthogonal_correctness():
    # Takes a known orthogonal pair and checks the function returns True
    assert is_orthogonal([1, 0, 0], [0, 1, 0]) == True

def test_is_non_orthogonal():
    # Takes a known non-orthogonal pair and checks the function returns False
    assert is_orthogonal([1, 2, 3], [4, 5, 6]) == False

# --- forward_pass test ---

@pytest.mark.parametrize("layers,vec", [
    ([[[1, 0], [0, 1]], [[2, 0], [0, 2]]], [3, 4]),
    ([[[1, 0], [1, 1]], [[2, 0], [3, 2]]], [4, 5]),
])
def test_forward_pass_correctness(layers, vec):
    # Checks forward pass result againt chained matrix multiplication in numpy
    numpy_result = np.array(vec, dtype=float)
    for layer in layers:
        numpy_result = np.array(layer) @ numpy_result
    assert np.allclose(forward_pass(layers, vec), numpy_result)


# --- attention_scores test ---

@pytest.mark.parametrize("query,keys", [
    ([1, 0, 1], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ([1, 1, 0], [[9, 2, 3], [4, 2, 5], [7, 7, 2]])
])
def test_attention_scores_correctness(query, keys):
    # Checks attention scores fuction against numpy equivelent
    assert np.allclose(attention_scores(query, keys), np.array(keys) @ query)

# --- transpose test ---

@pytest.mark.parametrize("m", [
    [[1, 2, 3], [4, 5, 6]],
    [[3, 2, 1], [3, 7, 8]],
    [[2, 4, 1], [6, 7, 7]]
])
def test_transpose_correctness(m):
    # Checks transpose function on matrices against numpy
    assert np.allclose(transpose(m), np.array(m).T)

# --- l1_norm test ---

@pytest.mark.parametrize("a", [
    [1, 2, 4, 7, 2, 8],
    [2, 3, 4, 1, 5, 6],
    [2, 3, 0, 0, 3, 4]
])
def test_l1_norm_correctness(a):
    # Checks l1 norm function against numpy
    assert np.allclose(l1_norm(a), np.linalg.norm(a, ord=1))

# --- l2_norm test ---

@pytest.mark.parametrize("a", [
    [1, 2, 4, 7, 2, 8],
    [2, 3, 4, 1, 5, 6],
    [2, 3, 0, 0, 3, 4]
])
def test_l2_norm_correctness(a):
    # Checks l2 norm function against numpy
    assert np.allclose(l2_norm(a), np.linalg.norm(a, ord=2))

# --- low_rank_approximation tests ---

@pytest.mark.parametrize("seed,shape,k", [
    (42, (5, 4), 2),
    (0,  (3, 3), 1),
    (7,  (6, 4), 3),
])
def test_low_rank_approximation_correctness(seed, shape, k):
    # Testing low rank approximation function against numpy
    rng = np.random.default_rng(seed)
    A = rng.random(shape)
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    expected = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    got = low_rank_approximation(U.tolist(), sigma.tolist(), Vt.tolist(), k)
    assert np.allclose(got, expected)

# --- mat_mult test ---

@pytest.mark.parametrize("A,B", [
    ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]),   # 2x3 @ 3x2
    ([[1, 0], [0, 1]], [[5, 6], [7, 8]]),                    # identity
])
def test_mat_mult_correctness(A, B):
    assert np.allclose(mat_mult(A, B), np.array(A) @ np.array(B))

# --- error path tests ---

def test_cosine_similarity_zero_vector():
    # Checks a zero vector raises ZeroDivisionError from the zero magnitude
    with pytest.raises(ZeroDivisionError):
        cosine_similarity([1, 2, 3], [0, 0, 0])

def test_cosine_similarity_length_mismatch():
    # Checks mismatched lengths raise ValueError from the underlying dot_product
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])

def test_magnitude_length_is_handled():
    # Checks magnitude of a single vector never raises on valid input
    assert magnitude([3, 4]) == pytest.approx(5.0)

def test_project_data_length_mismatch():
    # Checks a row shorter than the principal component raises ValueError
    with pytest.raises(ValueError):
        project_data([[1, 2]], [1, 2, 3])

def test_l1_norm_empty():
    # Checks the l1 norm of an empty vector is zero
    assert l1_norm([]) == 0

def test_l2_norm_empty():
    # Checks the l2 norm of an empty vector is zero
    assert l2_norm([]) == 0
