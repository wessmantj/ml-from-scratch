# birthday problem
n = 1
for n in range(365):
    product = 1
    for i in range(1, n + 1):
        product = product * (365 - i) / 365
    prob = 1 - product
    print(n)
    print(prob)
    if prob >= 0.5:
        break


# takes floats, returns posterior.

def bayes(prior, likelihood, evidence):
    pass
def cdf(x, distribution):
    pass

