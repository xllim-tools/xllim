.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>


This page describes the :ref:`GLLiM <gllim-class>` main methods.


.. _main-methods:

Main methods
------------

.. _initialize-method:

    .. method:: initialize(t, y, gllim_em_iteration, gllim_em_floor, gmm_kmeans_iteration, gmm_em_iteration, gmm_floor, nb_experiences, seed *= None*, verbose = 1)

        Initialize the GLLiM model with given data and parameters.

        :param ndarray t: Input matrix `t` with shape (N, L).
        :param ndarray y: Input matrix `y` with shape (N, D).
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

        :param ndarray x: Input matrix `x` with shape (N, L).
        :param ndarray y: Input matrix `y` with shape (N, D).
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

        :param ndarray x: Input matrix `x` with shape (N_obs, L).
        :param ndarray, optional x_incertitude: Uncertainty in `x` with shape (N_obs, L).
        :returns: An instance of `PredictionResult` containing the direct densities.


.. _inverse-densities-method:

    .. method:: inverseDensities(y, y_incertitude = 0, K_merged = 0, merging_threshold = 1e-10, verbose = 0)

        Compute the inverse densities given input matrix `y` and its uncertainties.

        :param ndarray y: Input matrix `y` with shape (N_obs, D).
        :param ndarray, optional y_incertitude: Uncertainty in `y` with shape (N_obs, D).
        :param int, optional K_merged: Merged the full GMM (K components) into K_merged gaussian components.
        :param float, optional merging_threshold: Threshold on the merged GMM weights. Gaussian component with a weight below this threshold are ignored.
        :param int verbose: Verbosity level (default is 0).
        :returns: An instance of `PredictionResult` containing the inverse densities.


.. _get-insights-method:

    .. method:: getInsights()

        Returns ann Insights structure with informations about initialisation and training time, log-likelihood and arguments.

        :returns: An instance of `Insights` containing total initialisation and trining time, training log-likelihood, initialisation specific infirmation and training specific information.
