.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>


This page describes the :ref:`GLLiM <gllim-class>` structures.



.. _structures:

Structures
----------


.. _gllim-parameters-struct:

.. class:: GLLiMParameters(L, D, K)

    A template structure representing the parameters of the GLLiM model.

    :param int L: The dimension of the model input (number of features) corresponding to the low-dimensional value.
    :param int D: The dimension of the model output corresponding to the high-dimensional value.
    :param int K: The number of affine transformations, corresponding to the number of Gaussian distributions in the mixture.

    .. attribute:: Pi
        :type: rowvec

        A row vector of size K containing the weights of the Gaussian distributions in the mixture.

    .. attribute:: A
        :type: ndarray of shape ()

        A ndarray of shape (D, L, K) representing the parameters of the affine transformations.
    
    .. attribute:: B
        :type: ndarray of shape ()

        A matrix of size (D, K) representing additional model parameters.

    .. attribute:: C
        :type: ndarray of shape ()

        A matrix of size (L, K) containing the means of the mixture of Gaussian distributions that define the low-dimensional data.

    .. attribute:: Gamma
        :type: ndarray

        A vector of size K containing the covariance matrices (L, L) of the mixture of Gaussian distributions that define the low-dimensional data.
            - In the case of Full covariance matrix (*gamma_type = `'full'`*), Gamma is of shape (K,L,L).
            - In the case of Diagonal covariance matrix (*gamma_type = `'diag'`*), Gamma is of shape (K,L) with Gamma[k] representing the variances vector of the k^{th} gaussian.
            - In the case of Isotropic covariance matrix (*gamma_type = `'iso'`*), Gamma is of shape (K) with Gamma[k] representing the unique variance of the k^{th} gaussian.

    .. attribute:: Sigma
        :type: ndarray

        A vector of size K containing the covariance matrices (D, D) of the mixture of Gaussian distributions that define the high-dimensional data.
            - In the case of Full covariance matrix (*gamma_type = `'full'`*), Sigma is of shape (K,L,L).
            - In the case of Diagonal covariance matrix (*gamma_type = `'diag'`*), Sigma is of shape (K,L) with Sigma[k] representing the variances vector of the k^{th} gaussian.
            - In the case of Isotropic covariance matrix (*gamma_type = `'iso'`*), Sigma is of shape (K) with Sigma[k] representing the unique variance of the k^{th} gaussian.


    .. note::

        For more detailed information on these parameters, refer to the formula in the paper: "High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables" by Antoine Deleforge, Florence Forbes, and Radu Horaud, published in *Statistics and Computing* 25(5): 893-911, September 2015.


.. _gllim-constraints-struct:

.. class:: GLLiMConstraints(gamma_type, sigma_type)

    A structure representing the constraints on the covariance matrices in the GLLiM model.

    :param str gamma_type: The type of the Gamma covariance matrix. It can be one of the following:
        - `'full'`: Full covariance matrix.
        - `'diag'`: Diagonal covariance matrix.
        - `'iso'`: Isotropic covariance matrix.
    :param str sigma_type: The type of the Sigma covariance matrix. It can be one of the following:
        - `'full'`: Full covariance matrix.
        - `'diag'`: Diagonal covariance matrix.
        - `'iso'`: Isotropic covariance matrix.

    .. attribute:: gamma_type
        :type: const std::string

        The Gamma covariance matrix type, indicating the structure of the covariance matrix in the low-dimensional space.

    .. attribute:: sigma_type
        :type: const std::string

        The Sigma covariance matrix type, indicating the structure of the covariance matrix in the high-dimensional space.


.. _prediction-result-struct:

.. class:: PredictionResult(N_obs, D, K)

    This structure combines the results from both the mean prediction and center prediction.

    :param int N_obs: Number of observations.
    :param int D: Dimensionality of each observation.
    :param int K: Number of components in the GMM.

    .. attribute:: meanPredResult
        :type: MeanPredictionResult

        The result of the mean prediction.

    .. attribute:: centerPredResult
        :type: CenterPredictionResult

        The result of the center prediction.


.. _mean-prediction-result-struct:

.. class:: MeanPredictionResult(N_obs, D, K)

    This structure holds the results of the mean predictions for a Gaussian Mixture Model (GMM).

    :param int N_obs: Number of observations.
    :param int D: Dimensionality of each observation.
    :param int K: Number of components in the GMM.

    .. attribute:: mean
        :type: ndarray of shape ()

        The mean of the GMM prediction (N_obs, D).

    .. attribute:: variance
        :type: ndarray of shape ()

        The variance of the GMM prediction (N_obs, D, D).

    .. attribute:: gmm_weights
        :type: ndarray of shape ()

        The weights of the components of the GMM (N_obs, K).

    .. attribute:: gmm_means
        :type: ndarray of shape ()

        The means of each component in the GMM (N_obs, D, K).

    .. attribute:: gmm_covs
        :type: ndarray of shape ()

        The covariance matrices of each component in the GMM (D, D, K).


.. _center-prediction-result-struct:

.. class:: CenterPredictionResult

    This structure holds the results of the center predictions for a Gaussian Mixture Model (GMM).

    .. attribute:: weights
        :type: vec

        The weights of the centers.

    .. attribute:: means
        :type: ndarray of shape ()

        The centers that represent the predictions.

    .. attribute:: covs
        :type: ndarray of shape ()

        The covariance matrices of the centers.



.. _insights-struct:

.. class:: Insights(time, log_likelihood, initialisation, training)

    This structure holds combined insights from both the initialization and training phases.

    :param datetime.timedelta time: The total time associated with the insights.
    :param vec log_likelihood: The log-likelihood values associated with the model.
    :param InitialisationInsights initialisation: Insights from the initialization phase.
    :param TrainingInsights training: Insights from the training phase.

    .. attribute:: time
        :type: datetime.timedelta

        The total time associated with the insights.

    .. attribute:: log_likelihood
        :type: vec

        The log-likelihood values associated with the model.

    .. attribute:: initialisation
        :type: InitialisationInsights

        Insights from the initialization phase.

    .. attribute:: training
        :type: TrainingInsights

        Insights from the training phase.


.. _initialisation-insights-struct:

.. class:: InitialisationInsights(time, start_time, end_time, N_obs, gllim_em_iteration, gllim_em_floor, gmm_kmeans_iteration, gmm_em_iteration, gmm_floor, nb_experiences)

    This structure holds insights into the initialization phase of the model training.

    :param datetime.timedelta time: The total time taken for initialization.
    :param datetime.datetime start_time: The start time of the initialization.
    :param datetime.datetime end_time: The end time of the initialization.
    :param int N_obs: Number of observations.
    :param int gllim_em_iteration: Number of GLLiM EM iterations during initialization.
    :param float gllim_em_floor: The floor value for GLLiM EM.
    :param int gmm_kmeans_iteration: Number of GMM k-means iterations.
    :param int gmm_em_iteration: Number of GMM EM iterations during initialization.
    :param float gmm_floor: The floor value for GMM.
    :param int nb_experiences: Number of experiences considered during initialization.

    .. attribute:: time
        :type: datetime.timedelta

        The total time taken for initialization.

    .. attribute:: start_time
        :type: datetime.datetime

        The start time of the initialization.

    .. attribute:: end_time
        :type: datetime.datetime

        The end time of the initialization.

    .. attribute:: N_obs
        :type: int

        Number of observations.

    .. attribute:: gllim_em_iteration
        :type: int

        Number of GLLiM EM iterations during initialization.

    .. attribute:: gllim_em_floor
        :type: float

        The floor value for GLLiM EM.

    .. attribute:: gmm_kmeans_iteration
        :type: int

        Number of GMM k-means iterations.

    .. attribute:: gmm_em_iteration
        :type: int

        Number of GMM EM iterations during initialization.

    .. attribute:: gmm_floor
        :type: float

        The floor value for GMM.

    .. attribute:: nb_experiences
        :type: int

        Number of experiences considered during initialization.


.. _training-insights-struct:

.. class:: TrainingInsights(time, start_time, end_time, N_obs, max_iteration, ratio_ll, floor)

    This structure holds insights into the training phase of the model.

    :param datetime.timedelta time: The total time taken for training.
    :param datetime.datetime start_time: The start time of the training.
    :param datetime.datetime end_time: The end time of the training.
    :param int N_obs: Number of observations.
    :param int max_iteration: The maximum number of iterations during training.
    :param float ratio_ll: The ratio of log-likelihood improvement.
    :param float floor: The floor value used during training.

    .. attribute:: time
        :type: datetime.timedelta

        The total time taken for training.

    .. attribute:: start_time
        :type: datetime.datetime

        The start time of the training.

    .. attribute:: end_time
        :type: datetime.datetime

        The end time of the training.

    .. attribute:: N_obs
        :type: int

        Number of observations.

    .. attribute:: max_iteration
        :type: int

        The maximum number of iterations during training.

    .. attribute:: ratio_ll
        :type: float

        The ratio of log-likelihood improvement.

    .. attribute:: floor
        :type: float

        The floor value used during training.

