.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>


This page describes the :ref:`GLLiM <gllim-class>` main methods.


.. _gllim-main-methods:

Main methods
------------

.. _initialize-method:

    .. method:: initialize(t, y, gllim_em_iteration, gllim_em_floor, gmm_kmeans_iteration, gmm_em_iteration, gmm_floor, nb_experiences, seed *= None*, verbose = 1)

        Initialize the GLLiM model with given data and parameters.

        :param ndarray of shape (L, N) t: Input matrix `t` with shape (L, N).
        :param ndarray of shape (D, N) y: Input matrix `y` with shape (D, N).
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

        :param ndarray of shape (L, N) x: Input matrix `x` with shape (L, N).
        :param ndarray of shape (D, N) y: Input matrix `y` with shape (D, N).
        :param int max_iteration: Maximum number of iterations.
        :param float ratio_ll: Ratio for log-likelihood convergence.
        :param float floor: Floor value for the training process.
        :param int verbose: Verbosity level (default is 1).


.. _train-jgmm-method:

    .. method:: trainJGMM(x, y, kmeans_iteration, em_iteration, floor, verbose = 1);

        Train the GLLiM model with given data and parameters. A classic GMM training is applied on the equivalent joint-GMM to GLLiM.
        The algorithm is provided by the Armadillo library. Check out the corresponding `Armadillo documentation <https://arma.sourceforge.net/docs.html#learn>`_ 
        for more details. This option is only available whith (*gamma_type* = 'full', *sigma_type* = 'full') constraints. The training 
        is equivalent and faster than the GLLiM-EM algorithm.

        :param ndarray of shape (L, N) x: Input matrix `x` with shape (L, N).
        :param ndarray of shape (D, N) y: Input matrix `y` with shape (D, N).
        :param int kmeans_iteration: The number of iterations of the k-means algorithm.
        :param int em_iteration: The number of iterations of the EM algorithm.
        :param float floor: The variance floor (smallest allowed value) for the diagonal covariances; setting this to a small non-zero value can help with convergence and/or better quality parameter estimates.
        :param int verbose: Verbosity level (default is 1).


.. _get-inverse-method:

    .. method:: getInverse()

        Get the inverse parameters of the GLLiM model.

        :returns: (*GLLiMParameters*) An instance of :ref:`GLLiMParameters <gllim-parameters-struct>` containing the inverse parameters.


.. _direct-densities-method:

    .. method:: directDensities(x, x_incertitude = 0)

        Compute the direct densities given input matrix `x` and its uncertainties.

        :param ndarray of shape (L, N_obs) x: Input matrix `x` with shape (L, N_obs).
        :param ndarray of shape (L, N_obs), optional x_incertitude: Uncertainty in `x` with shape (L, N_obs).
        :returns: (*PredictionResult*) An instance of :ref:`PredictionResult <prediction-result-struct>` containing the direct densities.


.. _inverse-densities-method:

    .. method:: inverseDensities(y, y_incertitude = 0, K_merged = 0, merging_threshold = 1e-10, verbose = 0)

        Compute the inverse densities given input matrix `y` and its uncertainties.

        :param ndarray of shape (D, N_obs) y: Input matrix `y` with shape (D, N_obs).
        :param ndarray of shape (D, N_obs), optional y_incertitude: Uncertainty in `y` with shape (D, N_obs).
        :param int, optional K_merged: Merged the full GMM (K components) into K_merged gaussian components.
        :param float, optional merging_threshold: Threshold on the merged GMM weights. Gaussian component with a weight below this threshold are ignored.
        :param int verbose: Verbosity level (default is 0).
        :returns: (*PredictionResult*) An instance of :ref:`PredictionResult <prediction-result-struct>` containing the inverse densities.


.. _get-insights-method:

    .. method:: getInsights()

        Returns an Insights structure with informations about initialisation and training time, log-likelihood and arguments.

        :returns: (*Insights*) An instance of :ref:`Insights <insights-struct>` containing total initialisation and trining time, training log-likelihood, initialisation specific infirmation and training specific information.
