import copy
import sys

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array

from dpmmlearn.probability import Prior
from dpmmlearn.utils import log_ewens_sampling_formula, pick_discrete

INT_MAX = sys.maxsize
MINUS_INFINITY = -float('inf')


class DPMM(BaseEstimator, ClusterMixin):
    """Dirichlet Process Mixture Model.  Using algorithm 2 from Neal (2000).

    Parameters
    ----------
    prob : Prior
        The Prior object for whatever model is being inferred.

    alpha : float
        DP concentration parameter.

    labels : array-likes of shape (n_samples, ), default=None
        Initial labels.

    thetas : list, default=None
        List of parameters of generation distribution.

    max_iter : int, default=500
        Maximum number of iterations of the DPMM algorithm to run.

    max_n_labels : int, default=100
        Maximum number of labels of the DPMM algorithm to run.
        Early stopping with iterations that exceed the max_n_labels.
        Initialization does not stop fitting, but iterates at least once.

    use_best_iter : bool, default=True
        Use the result when the likelihood was maximum during the fitting.

    patience : int, default=None
        Number of consecutive non improving epoch before early stopping.

    verbose : bool, default=True
        Controls the verbosity: the higher, the more messages.

        - True : the computation time for each fold and parameter candidate is displayed;
        - False : nothing is displayed;

    random_state : int, default=0
        Controls the randomness of Dirichlet Process.
    """

    def __init__(self,
                 prob,
                 alpha,
                 labels=None,
                 thetas=None,
                 max_iter=500,
                 max_n_labels=100,
                 use_best_iter=True,
                 patience=None,
                 verbose=True,
                 random_state=0):
        self.prob = prob
        self.alpha = float(alpha)
        self.labels = labels
        self.thetas = thetas
        self.max_iter = int(max_iter)
        self.max_n_labels = int(max_n_labels)
        self.use_best_iter = bool(use_best_iter)
        self.patience = patience
        self.verbose = bool(verbose)
        self.random_state = int(random_state)

        self.labels_ = None
        self.n_labels_ = None
        self.thetas_ = None

        self._p = None
        self._vmax = MINUS_INFINITY
        self._step = 0
        self._cur_iter = None

    def _check_params(self, X):
        n_samples = X.shape[0]
        # prior
        if not isinstance(self.prob, Prior):
            raise ValueError(
                f"prob must be an instance of Prior, got {self.prob} instead.")
        # alpha
        if self.alpha <= 0:
            raise ValueError(
                f"alpha must be > 0, got {self.alpha} instead.")
        # labels & thetas
        if (self.labels is not None) and (self.thetas is not None):
            if len(self.labels) != n_samples:
                raise ValueError(
                    f"length of labels must be equal n_samples, got {len(self.labels)} instead.")
            self.labels = np.array(self.labels, dtype=np.int32)
            if self.labels.ndim != 1:
                raise ValueError(
                    f"labels must be 1-d array, got {self.labels.ndim} instead.")
            if self.labels.min() != 0:
                raise ValueError(
                    f"minimum of labels must be 0, got {self.labels.min()} instead.")
            if not np.all(np.unique(self.labels) == np.arange(self.labels.min(), self.labels.max() + 1, dtype=np.int32)):
                raise ValueError(
                    "labels must be serial numbers starting from 0.")
            if not isinstance(self.thetas, list):
                raise ValueError(
                    "thetas must be list.")
            if len(self.thetas) != len(np.unique(self.labels)):
                raise ValueError(
                    f"length of labels must be equal number of labels, got {len(self.thetas)} instead.")
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter} instead.")
        # max_n_labels
        if self.max_n_labels <= 0:
            raise ValueError(
                f"max_n_labels must be > 0, got {self.max_n_labels} instead.")
        # patience
        if self.patience is not None:
            self.patience = int(self.patience)
            if self.patience <= 0:
                raise ValueError(
                    f"patience must be > 0, got {self.patience} instead.")
        # verbose
        if not isinstance(self.verbose, bool):
            raise ValueError(
                f"verbose must be bool, got {self.verbose} instead.")
        # random_state
        if not isinstance(self.random_state, int):
            raise ValueError(
                f"random_state must be int, got {self.random_state} instead.")
        if self.random_state < 0:
            raise ValueError(
                f"random_state must be >= 0, got {self.random_state} instead.")

    def _initialize(self, X):
        n_samples = X.shape[0]
        self.history_ = []

        # Probability for Gibbs sampling
        # First column is predicted probability of a new class with alpha
        # (n_samples, n_labels)
        self._p = self.alpha * self.prob.pred(X)[:, np.newaxis]

        if (self.labels is None) or (self.thetas is None):
            self.labels_ = np.zeros(n_samples, dtype=np.int32)
            self.n_labels_ = []
            self.thetas_ = []
        else:
            self.labels_ = self.labels.copy()
            self.n_labels_ = [np.sum(self.label == i).item() for i in range(self.label.max())]
            self.thetas_ = self.thetas

        self._cur_iter = 'init'
        self._best_iter = 0
        self._p_best = self._p.copy()
        self.labels_best = self.labels_.copy()
        self.n_labels_best = copy.deepcopy(self.n_labels_)
        self.thetas_best = copy.deepcopy(self.thetas_)

        np.random.seed(self.random_state)
        for i in range(n_samples):
            self._update_label_i(X, i)
        self._update_theta(X)

    def _end_update(self):
        is_early_stop = False
        self.history_.append(self._v_cur)

        if self._v_cur > self._vmax:
            self._vmax = self._v_cur
            self._best_iter = self._cur_iter

            self._p_best = self._p.copy()
            self.labels_best = self.labels_.copy()
            self.n_labels_best = copy.deepcopy(self.n_labels_)
            self.thetas_best = copy.deepcopy(self.thetas_)

            self._step = 0
        else:
            self._step += 1
            if (self.patience is not None) and (self._step > self.patience):
                is_early_stop = True
                if self.verbose:
                    print(f'iter={self._cur_iter} -- Reached patience')

        if len(self.n_labels_) >= self.max_n_labels:
            is_early_stop = True
            if self.verbose:
                print(f'iter={self._cur_iter} -- Reached max_n_labels')

        return is_early_stop

    def _commit_best_update(self):
        if self.use_best_iter:
            self._p = self._p_best.copy()
            self.labels_ = self.labels_best.copy()
            self.n_labels_ = copy.deepcopy(self.n_labels_best)
            self.thetas_ = copy.deepcopy(self.thetas_best)

    def _update_label_i(self, X, i):
        n_samples = X.shape[0]

        label = self._draw_new_label(i)
        if label == -1:  # If we selected to create a new cluster, then draw parameters for that cluster.
            self.n_labels_.append(1)
            self.labels_[i] = len(self.thetas_)
            if self.verbose:
                print(f"iter={self._cur_iter} -- New label created: {self.labels_[i]}")
            new_theta = self.prob.create_post(X[i]).sample()
            self.thetas_.append(new_theta)
            self._p = np.append(self._p, np.zeros((n_samples, 1), dtype=float), axis=1)
            self._p[i + 1:, -1] = self.prob.like1(X[i + 1:], new_theta)

        else:  # Otherwise just increment the count for the cloned cluster.
            self.labels_[i] = label
            self.n_labels_[label] = self.n_labels_[label] + 1

    def _del_label_i(self, i):
        label = self.labels_[i]
        # We're about to assign this point to a new cluster, so decrement current cluster count.
        self.n_labels_[label] = self.n_labels_[label] - 1
        # If we just deleted the last cluster member, then delete the cluster from self.phi
        if self.n_labels_[label] == 0:
            if self.verbose:
                print(f"iter={self._cur_iter} -- Label deleted: {label}")
            del self.thetas_[label]
            del self.n_labels_[label]
            # Need to decrement label numbers for labels greater than the one deleted...
            self.labels_[self.labels_ >= label] = self.labels_[self.labels_ >= label] - 1
            # And remove the corresponding probability column
            self._p = np.delete(self._p, label + 1, axis=1)

    def _update_theta(self, X):
        for i in range(len(self.thetas_)):
            idx = self.labels_ == i
            data = X[idx]
            new_theta = self.prob.create_post(data).sample()
            self.thetas_[i] = new_theta
            self._p[:, i + 1] = self.prob.like1(X, self.thetas_[i])

    def _draw_new_label(self, i):
        picked = pick_discrete(self._p[i] * np.append([1], self.n_labels_)) - 1
        return picked

    def _calc_posterior(self, X):
        v = log_ewens_sampling_formula(self.alpha, self.n_labels_)
        for i, theta in enumerate(self.thetas_):
            v = v + self.prob.lnprior(theta).item()
            idx = self.labels_ == i
            data = X[idx]
            v = v + self.prob.lnlikelihood(data, theta).item()
        self._v_cur = v

    def fit(self, X, y=None):
        """
        Perform clustering on `X` and returns cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by convention.
        """

        # validation
        X = check_array(X, ensure_2d=False)
        self._check_params(X)
        self._initialize(X)
        for i in range(self.max_iter):
            self._cur_iter = i + 1
            if self._update(X):
                break
        self._commit_best_update()
        self.labels_ = np.array(self.labels_)
        if self.verbose:
            print('=' * 10, 'Finished!', '=' * 10)
            print(f"best_iter={self._best_iter} -- n_labels: {len(self.n_labels_)}")
        return self

    def _update(self, X):
        n_samples = X.shape[0]
        for i in range(n_samples):
            self._del_label_i(i)
            self._update_label_i(X, i)
        self._update_theta(X)
        self._calc_posterior(X)
        return self._end_update()
