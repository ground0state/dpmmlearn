import numpy as np
from scipy.special import gamma, gammaln
import bisect


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def ewens_sampling_formula(alpha, count_list):
    n = np.sum(count_list)
    c = len(count_list)
    ascending_factorial = np.prod([i + alpha for i in range(n)])
    p = np.power(alpha, c) * np.prod([np.math.factorial(cnt - 1) for cnt in count_list]) / ascending_factorial
    return p.item()


def log_ewens_sampling_formula(alpha, count_list):
    n = np.sum(count_list)
    c = len(count_list)
    res = c * np.log(alpha)
    for i in range(c):
        param = count_list[i] - 1
        while param > 0:
            res += np.log(param)
            param -= 1
    for i in range(n):
        res -= np.log(i + alpha)
    return res.item()


# def vTmv(vec, mat=None, vec2=None):
#     """Multiply a vector transpose times a matrix times a vector.

#     @param vec  The first vector (will be transposed).
#     @param mat  The matrix in the middle.  Identity by default.
#     @param vec2 The second vector (will not be transposed.)  By default, the same as the vec.
#     @returns    Product.  Could be a scalar or a matrix depending on whether vec is a row or column
#                 vector.
#     """
#     if len(vec.shape) == 1:
#         vec = np.reshape(vec, [vec.shape[0], 1])
#     if mat is None:
#         mat = np.eye(len(vec))
#     if vec2 is None:
#         vec2 = vec
#     return np.dot(vec.T, np.dot(mat, vec2))


def gammad(d, nu_over_2):
    """D-dimensional gamma function."""
    nu = 2.0 * nu_over_2
    return np.pi**(d * (d - 1.) / 4) * \
        np.multiply.reduce([gamma(0.5 * (nu + 1 - i)) for i in range(d)])


def gammadln(d, nu_over_2):
    """D-dimensional log gamma function."""
    nu = 2.0 * nu_over_2
    return (d * (d - 1.) / 4) * np.log(np.pi) + \
        np.sum([gammaln(0.5 * (nu + 1 - i)) for i in range(d)])


def random_wish(dof, S, size=None):
    dim = S.shape[0]
    if size is None:
        x = np.random.multivariate_normal(np.zeros(dim), S, size=dof)
        return np.dot(x.T, x)
    else:
        if isinstance(size, int):
            size = (size,)
        out = np.empty(size + (dim, dim), dtype=np.float64)
        for ind in np.ndindex(size):
            x = np.random.multivariate_normal(np.zeros(dim), S, size=dof)
            out[ind] = np.dot(x.T, x)
        return out


def random_invwish(dof, invS, size=None):
    return np.linalg.inv(random_wish(dof, invS, size=size))


def pick_discrete(p):
    """Pick a discrete integer between 0 and len(p) - 1 with probability given by (normalized) p
    array.  Note that p array will be normalized here."""
    c = np.cumsum(p)
    c = c / c[-1]  # Normalize
    u = np.random.uniform()
    return bisect.bisect(c, u)
