# probability

> Probability and some statistic primitives implemented from scratch in pure Python, verified against NumPy.
 
## What this covers
 
 - Combinatorics (factorial, n-choose-k, birthday problem)
 - Bernoulli, binomial, multinomial, and Gaussian distributions
 - Bayes theorem & posterior updates
 - Expectation, variance, covariance, and correlation
 - Maximum Likelihood Estimation (MLE)
 - Law of Large Numbers & Central Limit Theorem (simulated)

## What clicked
 
MLE is where it clicked. I had been using cross-entropy loss in ML tutorials for months without understanding why it was the right choice. When I derived `mle_bernoulli` and `mle_gaussian` from first principles, I realized the loss function is just the negative log-likelihood of the distribution you're assuming your data comes from. If you assume Gaussian noise, you get MSE. If you assume Bernoulli outputs, you get binary cross-entropy. The loss function is already determined the moment you choose a probabilistic model.

Bayes also stopped feeling like a formula and started feeling like a process. Implementing `bayes_posterior`: you start with a prior distribution over hypotheses, observe some data, and the posterior is just the renormalized product. Every step has a name and a role. It clarified a lot of language I'd seen in ML papers like "prior," "posterior," "evidence".

## Why it matters for what's ahead
 
The Gaussian distribution and MLE show up in almost every model I'm about to build. Also, when I get to neural network training, the loss function will just be a log-likelihood I've already computed by hand. Covariance and correlation reappear in PCA, where the goal is literally to find the directions of maximum variance in a covariance matrix. Much more as well.

## Files
 
- `probability.py` — contains manually implemented functions
- `verify.py` — checks `probability.py` against its NumPy equivalent
- `probability-notes.pdf` — full notes from the source material

## How to run
 
```bash
python probability.py
python verify.py
```
 
## Status
 
> Implementations done and verified. Synthesis completed.