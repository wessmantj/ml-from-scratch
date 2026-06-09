import math
# returns the probability that at least two of n people share a birthday.
def birthday_problem(n):
    product = 1
    for i in range(n):
        product *= (365 - i) / 365
    return 1 - product


def factorial(n):
    res = 1
    
    for i in range(1, n + 1):
        res *= i
    return res


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


def bernoulli_pmf(k, p):
    if k == 1:
        return p
    elif k == 0:
        return 1 - p
    else:
        return 0


def bernoulli_sample(k, n, p):
    if k < 0 or k > n:
        return 0
    
    combinations = n_choose_k(n, k)

    probability = combinations * (p ** k) * ((1 - p) ** (n - k))

    return probability


def binomial_pmf(k, n, p):
    if k < 0 or k > n:
        return 0
    
    combinations = n_choose_k(n, k)

    res = combinations * (p ** k) * ((1 - p) ** (n - k))

    return res


def gaussian_pdf(x, mu, sigma):
    denominator = sigma * (math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu)/ sigma) ** 2

    return ((1/(denominator)) * math.exp(exponent))

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
    for count, prob in (counts, probs):
        match *= (prob * counts)

    
    return (numerator / denominator) * match

def bayes(prior, likelihood, evidence): 
    pass

def bayes_posterior(priors, likelihood): 
    pass

def expectations(values, probs):
    pass

def variance(values, probs):
    pass

def sample_mean(data):
    pass

def sample_variance(data, ddof):
    pass

def covariance(x, y):
    pass

def correlation(x, y):
    pass

def log_likelihood(data, dist, params):
    pass

def mle_bernoulli(data):
    pass

def mle_gaussian(data):
    pass

def simulate_lln(p, max_n):
    pass

def simulate_clt(sampler, num_samples, sample_size):
    pass



