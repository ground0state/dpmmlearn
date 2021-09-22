dpmmlearn
============

|image0| |image1| 

dpmmlearn is a algorithms for Dirichlet Process Mixture Model.


Dependencies
------------------------

The required dependencies to use dpmmlearn are,

- scikit-learn
- numpy
- scipy

You also need matplotlib, seaborn to run the demo and pytest to run the tests.

install
------------

.. code:: bash

    pip install dpmmlearn


USAGE
------------

We have posted a usage example in the github's demo folder.

Multi-dimensional Mixed Gaussian model by Dirichlet process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from dpmmlearn import DPMM
    from dpmmlearn.probability import NormInvWish


    mu_0 = 0.0
    kappa_0 = 1.0
    Lam_0 = np.eye(2) * 10
    nu_0 = 2
    alpha = 3.0

    prob = NormInvWish(mu_0, kappa_0, Lam_0, nu_0)
    model = DPMM(prob, alpha, max_iter=300, random_state=0, verbose=True)

    model.fit(X)

    labels = model.labels_



One-dimensional Mixed Gaussian model by Dirichlet process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from dpmmlearn import DPMM
    from dpmmlearn.probability import NormInvChi2


    def sample_pdf(x, mu_list, var_list, portion):
        y = np.zeros_like(x)
        for mu, var, num in zip(mu_list, var_list, portion):
            y_ = np.exp(-0.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)
            y += num/sum(portion) * y_
        return y


    mu_0 = 0.3
    kappa_0 = 0.1
    sigsqr_0 = 0.1
    nu_0 = 1.0
    alpha = 1.0

    prob = NormInvChi2(mu_0, kappa_0, sigsqr_0, nu_0)
    model = DPMM(prob, alpha, max_iter=500, random_state=0, verbose=True)

    model.fit(X)

    mu_ = [mu for mu, var in model.thetas_]
    var_ = [var for mu, var in model.thetas_]
    portion_ = [k/sum(model.n_labels_) for k in model.n_labels_]

    x = np.arange(-2, 2, 0.01)
    y_pred = sample_pdf(x, mu_, var_, portion_)


One-dimensional variance estimation of mixed Gaussian model by Dirichlet process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from dpmmlearn import DPMM
    from dpmmlearn.probability import InvGamma


    def result_pdf(x, thetas, n_labels, mu):
        y = np.zeros_like(x)
        for theta, n_label in zip(thetas, n_labels):
            var = theta
            y_ = np.exp(-0.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)
            y += n_label/sum(n_labels) * y_
        return y


    mu = 0.0
    alpha = 1.0
    beta = 100.0
    dp_alpha = 0.1

    prob = InvGamma(alpha, beta, mu)
    model = DPMM(prob, dp_alpha, max_iter=500, random_state=0, verbose=True)

    model.fit(X)

    y_pred = result_pdf(x, model.thetas_, model.n_labels_, mu)


License
------------

This code is licensed under MIT License.

Test
------------

.. code:: python

    python setup.py test

.. |image0| image:: https://img.shields.io/badge/dynamic/json.svg?label=version&colorB=5f9ea0&query=$.version&uri=https://raw.githubusercontent.com/ground0state/dpmmlearn/main/package.json&style=plastic
.. |image1| image:: https://static.pepy.tech/personalized-badge/dpmmlearn?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
 :target: https://pepy.tech/project/dpmmlearn