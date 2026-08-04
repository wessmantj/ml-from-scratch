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

# --- project_data tests --- 