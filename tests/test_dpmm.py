from dpmmlearn import DPMM
from dpmmlearn import probability
import gmm

# Some common test data:
mu = [-0.5, 0.0, 0.7]  # means
V = [0.02, 0.03, 0.1]  # variances
p = [0.25, 0.4, 0.35]  # proportions
model = gmm.GMM([gmm.GaussND(mu0, V0) for mu0, V0 in zip(mu, V)], p)
data = model.sample(size=100)


def test_dpmm_GaussianMeanKnownVariance():
    mu_0 = 0.0
    sigsqr_0 = 1.0
    sigsqr = 0.05
    cp = probability.GaussianMeanKnownVariance(mu_0, sigsqr_0, sigsqr)
    alpha = 1.0

    # Just checking that we can construct a DP and update it.
    dp = DPMM(cp, alpha, max_iter=100)


def test_dpmm_NormInvChi2():
    mu_0 = 0.3
    kappa_0 = 0.1
    sigsqr_0 = 1.0
    nu_0 = 0.1
    cp = probability.NormInvChi2(mu_0, kappa_0, sigsqr_0, nu_0)
    alpha = 10.0

    # Just checking that we can construct a DP and update it.
    dp = DPMM(cp, alpha, max_iter=100)
