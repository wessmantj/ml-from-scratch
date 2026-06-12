import math
import random
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

TOL = 1e-9

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


# 1. factorial vs math.factorial
check("factorial", factorial(10), math.factorial(10))
check("factorial (0)", factorial(0), math.factorial(0))

# 2. n_choose_k vs math.comb
check("n_choose_k", n_choose_k(10, 3), math.comb(10, 3))
check("n_choose_k (k=0)", n_choose_k(5, 0), math.comb(5, 0))
check("n_choose_k (k=n)", n_choose_k(5, 5), math.comb(5, 5))

# 3. birthday_problem — manual: 1 - prod((365-i)/365 for i in range(n))
n = 23
expected = 1 - np.prod([(365 - i) / 365 for i in range(n)])
check("birthday_problem (n=23)", birthday_problem(n), expected)

# 4. bernoulli_pmf vs direct formula
check("bernoulli_pmf (k=1)", bernoulli_pmf(1, 0.7), 0.7)
check("bernoulli_pmf (k=0)", bernoulli_pmf(0, 0.7), 0.3)

# 5. bernoulli_sample vs C(n,k) * p^k * (1-p)^(n-k)
k, n, p = 3, 10, 0.4
check("bernoulli_sample", bernoulli_sample(k, n, p), math.comb(n, k) * p**k * (1-p)**(n-k))

# 6. binomial_pmf vs C(n,k) * p^k * (1-p)^(n-k)
check("binomial_pmf", binomial_pmf(3, 10, 0.4), math.comb(10, 3) * 0.4**3 * 0.6**7)
check("binomial_pmf (k=0)", binomial_pmf(0, 5, 0.3), math.comb(5, 0) * 0.3**0 * 0.7**5)

# 7. binomial_mean / binomial_variance vs n*p and n*p*(1-p)
check("binomial_mean", binomial_mean(10, 0.4), 10 * 0.4)
check("binomial_variance", binomial_variance(10, 0.4), 10 * 0.4 * 0.6)

# 8. gaussian_pdf vs (1/(sigma*sqrt(2*pi))) * exp(-0.5*((x-mu)/sigma)^2)
def ref_gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

check("gaussian_pdf", gaussian_pdf(1.5, 0.0, 1.0), ref_gaussian_pdf(1.5, 0.0, 1.0))
check("gaussian_pdf (off-center)", gaussian_pdf(2.0, 1.0, 2.0), ref_gaussian_pdf(2.0, 1.0, 2.0))

# 9. gaussian_cdf vs 0.5 * (1 + erf((x-mu) / (sigma*sqrt(2))))
def ref_gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

check("gaussian_cdf", gaussian_cdf(1.5, 0.0, 1.0), ref_gaussian_cdf(1.5, 0.0, 1.0))
check("gaussian_cdf (off-center)", gaussian_cdf(0.0, 1.0, 2.0), ref_gaussian_cdf(0.0, 1.0, 2.0))

# 10. multinomial_pmf — manual: n! / prod(ci!) * prod(pi^ci)
counts = [2, 3, 1]
probs = [0.2, 0.5, 0.3]
n = sum(counts)
ref_multi = (math.factorial(n) / np.prod([math.factorial(c) for c in counts])) * np.prod([p**c for p, c in zip(probs, counts)])
check("multinomial_pmf", multinomial_pmf(counts, probs), ref_multi)

# 11. bayes vs (likelihood * prior) / evidence
prior, likelihood, evidence = 0.3, 0.8, 0.5
check("bayes", bayes(prior, likelihood, evidence), (likelihood * prior) / evidence)

# 12. bayes_posterior — manual normalization
priors = [0.3, 0.7]
likelihoods = [0.9, 0.2]
unnorm = [p * l for p, l in zip(priors, likelihoods)]
ev = sum(unnorm)
check("bayes_posterior", bayes_posterior(priors, likelihoods), [u / ev for u in unnorm])

# 13. expectations vs np.average with weights
values = [1, 2, 3, 4]
probs = [0.1, 0.4, 0.3, 0.2]
check("expectations", expectations(values, probs), np.average(values, weights=probs))

# 14. variance vs E[X^2] - E[X]^2
ex2 = sum(v**2 * p for v, p in zip(values, probs))
ex = sum(v * p for v, p in zip(values, probs))
check("variance", variance(values, probs), ex2 - ex**2)

# 15. sample_mean vs np.mean
data = [2.0, 4.0, 6.0, 8.0, 10.0]
check("sample_mean", sample_mean(data), np.mean(data))

# 16. sample_variance vs np.var with matching ddof
check("sample_variance (ddof=0)", sample_variance(data, ddof=0), np.var(data, ddof=0))
check("sample_variance (ddof=1)", sample_variance(data, ddof=1), np.var(data, ddof=1))

# 17. covariance vs np.cov (ddof=0, population covariance)
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 5.0, 4.0, 5.0]
check("covariance", covariance(x, y), np.cov(x, y, ddof=0)[0, 1])

# 18. correlation vs np.corrcoef (Pearson r is ddof-invariant)
check("correlation", correlation(x, y), np.corrcoef(x, y)[0, 1])

# 19. mle_bernoulli vs np.mean
bern_data = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
check("mle_bernoulli", mle_bernoulli(bern_data), np.mean(bern_data))

# 20. mle_gaussian vs np.mean and np.std (ddof=0)
gauss_data = [2.1, 2.5, 3.0, 2.8, 2.3]
mu, sigma = mle_gaussian(gauss_data)
check("mle_gaussian (mu)", mu, np.mean(gauss_data))
check("mle_gaussian (sigma)", sigma, np.std(gauss_data, ddof=0))

# 21. log_likelihood (bernoulli) vs manual sum of log probs
ll_data = [1, 0, 1, 1, 0]
p = 0.6
expected_ll = sum(math.log(p if x == 1 else 1 - p) for x in ll_data)
check("log_likelihood (bernoulli)", log_likelihood(ll_data, "bernoulli", p), expected_ll)

# 22. log_likelihood (gaussian) vs manual sum using ref_gaussian_pdf
ll_data = [1.0, 2.0, 3.0]
mu, sigma = 2.0, 1.0
expected_ll = sum(math.log(ref_gaussian_pdf(x, mu, sigma)) for x in ll_data)
check("log_likelihood (gaussian)", log_likelihood(ll_data, "gaussian", (mu, sigma)), expected_ll)

# 23. simulate_lln — check length and convergence near true p
random.seed(42)
proportions = simulate_lln(0.5, 10_000)
check("simulate_lln (length)", len(proportions), 10_000)
check("simulate_lln (converges near p)", abs(proportions[-1] - 0.5) < 0.05, True)

# 24. simulate_clt — check length and that sample means converge near E[X]=0.5
random.seed(42)
sample_means = simulate_clt(lambda: random.uniform(0, 1), 2_000, 50)
check("simulate_clt (length)", len(sample_means), 2_000)
check("simulate_clt (mean near 0.5)", abs(np.mean(sample_means) - 0.5) < 0.02, True)
