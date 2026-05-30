# returns the probability that at least two of n people share a birthday.
def birthday_problem(n):
    product = 1
    for i in range(1, n + 1):
        product *= (365 - i) / 365
    return 1 - product

# n = 22 is over 50%

def bayes(prior, likelihood, evidence): 
    pass
def cdf(x, distribution):
    pass

