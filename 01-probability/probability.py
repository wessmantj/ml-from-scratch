import math
import random
from typing import Tuple, List

# returns the probability that at least two of n people share a birthday.
def birthday_problem(n):
    product = 1
    for i in range(n):
        product *= (365 - i) / 365
    return 1 - product

# returns the factorial for some value n
def factorial(n):
    res = 1
    
    for i in range(1, n + 1):
        res *= i
    return res

# returns the end product of some n choose some k as an int
def n_choose_k(n, k):
    if k < 0 or k > n:
        return 0
    
    k = min(k, n-k)

    numerator = 1
    denominator = 1

    for i in range(k):
        numerator *= (n - i)
        denominator *= (i + 1)
    
    return numerator // denominator

# bernoulli's formula, returns either 1 or 9, yes or no, etc.
def bernoulli_pmf(k, p):
    if k == 1:
        return p
    elif k == 0:
        return 1 - p
    else:
        return 0

# bernoullis formula reworked for getting k sucesses in n trials
def bernoulli_sample(k, n, p):
    if k < 0 or k > n:
        return 0
    
    combinations = n_choose_k(n, k)

    probability = combinations * (p ** k) * ((1 - p) ** (n - k))

    return probability

#  give the probability of k succcesses out of n independent trials
def binomial_pmf(k, n, p):
    if k < 0 or k > n:
        return 0
    
    combinations = n_choose_k(n, k)

    res = combinations * (p ** k) * ((1 - p) ** (n - k))

    return res

# mean
def binomial_mean(n, p):
    return n * p

# variande
def binomial_variance(n, p):
    return n * p * (1 - p)


#  gives the height of the gaussian curve at a given point x
def gaussian_pdf(x, mu, sigma):
    denominator = sigma * (math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu)/ sigma) ** 2

    return ((1/(denominator)) * math.exp(exponent))

#mcalculates the probability that a value will be less than or equal to x
def gaussian_cdf(x, mu, sigma):
    denominator = sigma * (math.sqrt(2))
    
    return 0.5 * (1 + math.erf((x - mu) / denominator))


def multinomial_pmf(counts, probs):

    n = sum(counts)

    numerator = factorial(n)

    denominator = 1
    for count in counts:
        denominator *= factorial(count)

    match = 1
    for count, prob in zip(counts, probs):
        match *= (prob ** count)


    return (numerator / denominator) * match


def bayes(prior, likelihood, evidence): 
    return (likelihood * prior) / evidence


def bayes_posterior(priors, likelihoods): 
    unnormalized = []

    for prior, likelihood in zip(priors, likelihoods):
        unnormalized.append(prior * likelihood)

    evidence = sum(unnormalized)

    posteriors = []

    for u in unnormalized:
        posteriors.append(u / evidence)

    return posteriors


def expectations(values, probs):
    mean = 0
    for value, prob in zip(values, probs):
        mean += value * prob
    return mean


def variance(values, probs):
    mean = expectations(values, probs)
    expectation_x_squared = 0
    for value, prob in zip(values, probs):
        expectation_x_squared += (value ** 2) * prob
    return expectation_x_squared - (mean ** 2)


def sample_mean(data: List):
    count = len(data)
    total = sum(data)
    return total / count


def sample_variance(data, ddof):
    mean = sample_mean(data)
    n = len(data)
    var = sum((x - mean) ** 2 for x in data) / (n - ddof)

    return var


def covariance(x, y):
    if len(x) != len(y):
        return 0
    mean_x = sample_mean(x)
    mean_y = sample_mean(y)
    n = len(x)
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    return cov

def correlation(x, y):
    cov = covariance(x, y)
    std_x = math.sqrt(sample_variance(x, ddof=0))
    std_y = math.sqrt(sample_variance(y, ddof=0))
    if std_x == 0 or std_y == 0:
        return 0
    return cov / (std_x * std_y)

def log_likelihood(data, dist, params):
    ll = 0
    if dist == "bernoulli":
        p = params
        for x in data:
            ll += math.log(bernoulli_pmf(x, p))
    elif dist == "gaussian":
        mu, sigma = params
        for x in data:
            ll += math.log(gaussian_pdf(x, mu, sigma))
    return ll

def mle_bernoulli(data):
    return sum(data) / len(data)

def mle_gaussian(data):
    mu = sample_mean(data)
    sigma_squared = sample_variance(data, ddof=0)
    return (mu, math.sqrt(sigma_squared))

def simulate_lln(p, max_n):
    running_sum = 0
    proportions = []
    for i in range(1, max_n + 1):
        outcome = 1 if random.random() < p else 0
        running_sum += outcome
        proportions.append(running_sum / i)
    return proportions

def simulate_clt(sampler, num_samples, sample_size):
    sample_means = []
    for _ in range(num_samples):
        sample = [sampler() for _ in range(sample_size)]
        sample_means.append(sample_mean(sample))
    return sample_means



