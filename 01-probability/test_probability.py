import pytest
import math
import numpy as np
from probability import (
    birthday_problem,
    factorial,
    n_choose_k,
    bernoulli_pmf,
    bernoulli_sample,
    binomial_mean,
    binomial_pmf,
    binomial_variance,
    gaussian_cdf,
    gaussian_pdf,
    mle_gaussian,
    multinomial_pmf,
    bayes,
    bayes_posterior,
    mle_bernoulli,
    expectations,
    variance,
    sample_mean,
    sample_variance,
    covariance,
    correlation,
    log_likelihood,
    simulate_clt,
    simulate_lln
)

# --- birthday_problem tests ---

def test_birthday_problem_over_fifty():
    # 23 is the smallest group where P(shared birthday) exceeds 50%
    assert birthday_problem(23) > 0.50
    assert birthday_problem(22) < 0.50

def test_birthday_problem_over_ninetynine():
    # 57 is the smallest group where P(shared birthday) exceeds 99%
    assert birthday_problem(57) > 0.99
    assert birthday_problem(56) < 0.99

# --- factorial test ---

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 20, 1002, 9404, 93])
def test_factorial_correctness(n):
    # Checks factorial function against math.factorial
    assert factorial(n) == math.factorial(n)

# --- n_choose_k tests ---

@pytest.mark.parametrize("n,k", [
    [10,3],
    [5, 0],
    [5, 5],
    [2, 8]
])
def test_n_choose_k_correctness(n, k):
    # Checks n_choose_k against math.comb
    assert n_choose_k(n, k) == math.comb(n, k)

# --- bernoulli_pmf tests ---

@pytest.mark.parametrize("k, p, expected", [
    (1, 0.3, 0.3),      # success returns p
    (0, 0.3, 0.7),      # failure returns 1 - p
    (2, 0.3, 0.0),      # impossible 
    (-1, 0.3, 0.0),     
    (1, 1.0, 1.0),      
    (1, 0.0, 0.0),      
])
def test_bernoulli_pmf_correctness(k, p, expected):
    assert bernoulli_pmf(k, p) == pytest.approx(expected)

@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_bernoulli_pmf_sums_to_one(p):
    # Every PMF's outcomes must sum to 1
    assert bernoulli_pmf(0, p) + bernoulli_pmf(1, p) == pytest.approx(1.0)
    
# --- bernoulli_sample tests ---

def test_bernoulli_sample_correctness():
    ...

# --- binomial_pmf tests ---

def test_binomial_pmf_correctness():
    ...

# --- binomial_mean tests ---

def test_binomial_mean_correctness():
    ...

# --- binomial_variance tests ---

def test_binomial_variance_correctness():
    ...

# --- gaussian_pdf tests ---

def test_gaussian_pdf_correctness():
    ...

# --- gaussian_cdf tests ---

def test_gaussian_cdf_correctness():
    ...

# --- multinomial_pmf tests ---

def test_multinomial_pmf_correctness():
    ...

# --- bayes tests ---

def test_bayes_correctness():
    ...

# --- bayes_posterior tests ---

def test_bayes_posterior_correctness():
    ...

# --- expectations tests ---

def test_expectations_correctness():
    ...

# --- variance tests ---

def test_variance_correctness():
    ...

# --- sample_mean test ---

def test_sample_mean_correctness():
    ...

# --- sample_variance test ---

def test_sample_variance_correctness():
    ...

# --- covariance tests ---

def test_covariance_correctness():
    ...

# --- correlation test ---

def test_correlation_correctness():
    ...

# --- log_likelihood tests ---

def test_log_likelihood_correctness():
    ...

# --- mle_bernoulli tests ---

def test_mle_bernoulli_correctness():
    ...

# --- mle_gaussian test ---

def test_mle_gaussian_correctness():
    ...

# --- simulate_lln tests ---

def test_simulate_lln_correctness():
    ...

# --- simulate_clt tests ---

def test_simulate_clt_correctness():
    ...


