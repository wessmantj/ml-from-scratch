# returns the probability that at least two of n people share a birthday.
def birthday_problem(n):
    product = 1
    for i in range(n):
        product *= (365 - i) / 365
    return 1 - product

# n = 22 is over 50%


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


print(n_choose_k(10, 3))
def bernoulli_pmf(k, p):
    pass

def bernoulli_sample(k, n, p):
    pass

def binomial_pmf(k, n, p):
    pass

def gaussian_pdf(x, mu, sigma):
    pass

def gaussian_cdf(x, mu, sigma):
    pass

def multinomial_pmf(counts, probs):
    pass

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



