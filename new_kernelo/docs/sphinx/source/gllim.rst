GLLiM Class
===========

.. class:: GLLiM (L, D, K, gamma_type, sigma_type )

    Gaussian Locally-Linear Model (GLLiM) for probabilistic modeling.

    :param int L: The latent space dimension.
    :param int D: The observed space dimension.
    :param int K: The number of Gaussian components.
    :param str gamma_type: The type of gamma parameter.
    :param str sigma_type: The type of sigma parameter.
    :returns: An instance of the GLLiM class.


:ref:`Main methods<main-methods>`
*********************************

+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| :ref:`initialize<initialize-method>` (*t*, *y*, *gllim_em_iteration*, *gllim_em_floor*, *gmm_kmeans_iteration*, *gmm_em_iteration*, *gmm_floor*, *nb_experiences*,*seed*, *verbose=1*) | Initialize the GLLiM model with given data and parameters.                  |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| :ref:`train <train-method>` (*x*, *y*, *max_iteration*, *ratio_ll*, *floor*, *verbose=1*)                                                                                              | Train the GLLiM model with given data and parameters.                       |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| :ref:`getInverse <get-inverse-method>` ()                                                                                                                                              | Get the inverse parameters of the GLLiM model.                              |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| :ref:`directDensities <direct-densities-method>` (*x*, *x_incertitude=0*)                                                                                                              | Compute the direct densities given input matrix `x` and its uncertainties.  |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| :ref:`inverseDensities <inverse-densities-method>` (*y*, *y_incertitude=0*)                                                                                                            | Compute the inverse densities given input matrix `y` and its uncertainties. |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+

:ref:`Getters<getters>`
***********************

+---------------------------------------------------+-----------------------------------------+
| :ref:`getDimensions <get-dimensions-method>` ()   | Get the dimensions of the GLLiM model.  |
+---------------------------------------------------+-----------------------------------------+
| :ref:`getConstraints <get-constraints-method>` () | Get the constraints of the GLLiM model. |
+---------------------------------------------------+-----------------------------------------+
| :ref:`getParams <get-params-method>` ()           | Get the parameters of the GLLiM model.  |
+---------------------------------------------------+-----------------------------------------+
| :ref:`getParamPi <get-param-pi-method>` ()        | Get the mixture coefficients `Pi`.      |
+---------------------------------------------------+-----------------------------------------+
| :ref:`getParamA <get-param-a-method>` ()          | Get the parameter matrix `A`.           |
+---------------------------------------------------+-----------------------------------------+
| :ref:`getParamB <get-param-b-method>` ()          | Get the parameter matrix `B`.           |
+---------------------------------------------------+-----------------------------------------+
| :ref:`getParamC <get-param-c-method>` ()          | Get the parameter matrix `C`.           |
+---------------------------------------------------+-----------------------------------------+
| :ref:`getParamGamma <get-param-gamma-method>` ()  | Get the gamma parameters.               |
+---------------------------------------------------+-----------------------------------------+
| :ref:`getParamSigma <get-param-sigma-method>` ()  | Get the sigma parameters.               |
+---------------------------------------------------+-----------------------------------------+

:ref:`Setters<setters>`
***********************

+-------------------------------------------------------+----------------------------------------+
| :ref:`setParams <set-params-method>` (theta)          | Set the parameters of the GLLiM model. |
+-------------------------------------------------------+----------------------------------------+
| :ref:`setParamPi <set-param-pi-method>` (Pi)          | Set the mixture coefficients `Pi`.     |
+-------------------------------------------------------+----------------------------------------+
| :ref:`setParamA <set-param-a-method>` (A)             | Set the parameter matrix `A`.          |
+-------------------------------------------------------+----------------------------------------+
| :ref:`setParamB <set-param-b-method>` (B)             | Set the parameter matrix `B`.          |
+-------------------------------------------------------+----------------------------------------+
| :ref:`setParamC <set-param-c-method>` (C)             | Set the parameter matrix `C`.          |
+-------------------------------------------------------+----------------------------------------+
| :ref:`setParamGamma <set-param-gamma-method>` (Gamma) | Set the gamma parameters.              |
+-------------------------------------------------------+----------------------------------------+
| :ref:`setParamSigma <set-param-sigma-method>` (Sigma) | Set the sigma parameters.              |
+-------------------------------------------------------+----------------------------------------+


.. _main-methods:

Main methods
------------

.. _initialize-method:

    .. method:: initialize(t, y, gllim_em_iteration, gllim_em_floor, gmm_kmeans_iteration, gmm_em_iteration, gmm_floor, nb_experiences, seed *= None*, verbose = 1)

        Initialize the GLLiM model with given data and parameters.

        :param ndarray t: Input matrix `t`.
        :param ndarray y: Input matrix `y`.
        :param int gllim_em_iteration: Number of EM iterations for GLLiM.
        :param float gllim_em_floor: Floor value for EM iterations in GLLiM.
        :param int gmm_kmeans_iteration: Number of k-means iterations for GMM.
        :param int gmm_em_iteration: Number of EM iterations for GMM.
        :param float gmm_floor: Floor value for EM iterations in GMM.
        :param int nb_experiences: Number of experiences.
        :param int seed: Random seed for initialization.
        :param int verbose: Verbosity level (default is 1).


.. _train-method:

    .. method:: train(x, y, max_iteration, ratio_ll, floor, verbose=1)

        Train the GLLiM model with given data and parameters.

        :param ndarray x: Input matrix `x`.
        :param ndarray y: Input matrix `y`.
        :param int max_iteration: Maximum number of iterations.
        :param float ratio_ll: Ratio for log-likelihood convergence.
        :param float floor: Floor value for the training process.
        :param int verbose: Verbosity level (default is 1).


.. _get-inverse-method:

    .. method:: getInverse()

        Get the inverse parameters of the GLLiM model.

        :returns: An instance of `GLLiMParameters` containing the inverse parameters.


.. _direct-densities-method:

    .. method:: directDensities(x, x_incertitude = 0)

        Compute the direct densities given input matrix `x` and its uncertainties.

        :param ndarray x: Input matrix `x`.
        :param ndarray, optional x_incertitude: Uncertainty in `x`.
        :returns: An instance of `PredictionResult` containing the direct densities.


.. _inverse-densities-method:

    .. method:: inverseDensities(y, y_incertitude = 0)

        Compute the inverse densities given input matrix `y` and its uncertainties.

        :param ndarray y: Input matrix `y`.
        :param ndarray, optional y_incertitude: Uncertainty in `y`.
        :returns: An instance of `PredictionResult` containing the inverse densities.


.. _getters:

Getters
-------

.. _get-dimensions-method:

    .. method:: getDimensions()

        Get the dimensions of the GLLiM model.

        :returns: A string describing the dimensions of the model.


.. _get-constraints-method:

    .. method:: getConstraints()

        Get the constraints of the GLLiM model.

        :returns: string A string describing the constraints of the model.


.. _get-params-method:

    .. method:: getParams()

        Get the parameters of the GLLiM model.

        :returns: An instance of `GLLiMParameters` containing the model parameters.


.. _get-param-pi-method:

    .. method:: getParamPi()

        Get the mixture coefficients `Pi`.

        :returns: A row vector of mixture coefficients.


.. _get-param-a-method:

    .. method:: getParamA()

        Get the parameter matrix `A`.

        :returns: A cube containing the parameter matrix `A`.


.. _get-param-b-method:

    .. method:: getParamB()

        Get the parameter matrix `B`.

        :returns: A matrix containing the parameter matrix `B`.


.. _get-param-c-method:

    .. method:: getParamC()

        Get the parameter matrix `C`.

        :returns: A matrix containing the parameter matrix `C`.


.. _get-param-gamma-method:

    .. method:: getParamGamma()

        Get the gamma parameters.

        :returns: A ndarray representing Gamma. The array is of shape (K) if Gamma constraint is "iso", of shape (K,L) if "diag" or of shape (K,L,L) if "full".


.. _get-param-sigma-method:

    .. method:: getParamSigma()

        Get the sigma parameters.

        :returns: A ndarray representing Sigma. The array is of shape (K) if Sigma constraint is "iso", of shape (K,D) if "diag" or of shape (K,D,D) if "full".



.. _setters:

Setters
-------

.. _set-params-method:

    .. method:: setParams(theta)

        Set the parameters of the GLLiM model.

        :param GLLiMParameters theta: The model parameters to set.


.. _set-param-pi-method:

    .. method:: setParamPi(Pi)

        Set the mixture coefficients `Pi`.

        :param ndarray Pi: The mixture coefficients to set.


.. _set-param-a-method:

    .. method:: setParamA(A)

        Set the parameter matrix `A`.

        :param ndarray A: The parameter matrix to set.


.. _set-param-b-method:

    .. method:: setParamB(B)

        Set the parameter matrix `B`.

        :param ndarray B: The parameter matrix to set.


.. _set-param-c-method:

    .. method:: setParamC(C)

        Set the parameter matrix `C`.

        :param ndarray C: The parameter matrix to set.


.. _set-param-gamma-method:

    .. method:: setParamGamma(Gamma)

        Set the gamma parameters.

        :param ndarray Gamma: The gamma parameters to set. Shape depends on Gamma constraints.


.. _set-param-sigma-method:

    .. method:: setParamSigma(Sigma)

        Set the sigma parameters.

        :param ndarray Sigma: The sigma parameters to set. Shape depends on Sigma constraints.
