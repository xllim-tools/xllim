.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>



.. _gllim-class:

GLLiM class overall
===================

.. class:: GLLiM (L, D, K, gamma_type, sigma_type )

    Gaussian Locally-Linear Model (GLLiM) for probabilistic modeling.

    :param int L: The latent space dimension.
    :param int D: The observed space dimension.
    :param int K: The number of Gaussian components.
    :param str gamma_type: The type of gamma parameter.
    :param str sigma_type: The type of sigma parameter.
    :returns: An instance of the GLLiM class.


    .. toctree::
        :hidden:

        gllim_main_methods
        gllim_getters
        gllim_setters
        gllim_structures

        
    :ref:`Main methods <main-methods>`
    ----------------------------------

    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :ref:`initialize<initialize-method>` (*t*, *y*, *gllim_em_iteration*, *gllim_em_floor*, *gmm_kmeans_iteration*, *gmm_em_iteration*, *gmm_floor*, *nb_experiences*,*seed*, *verbose=1*) | Initialize the GLLiM model with given data and parameters.                                                             |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :ref:`train <train-method>` (*x*, *y*, *max_iteration*, *ratio_ll*, *floor*, *verbose=1*)                                                                                              | Train the GLLiM model with given data and parameters.                                                                  |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :ref:`getInverse <get-inverse-method>` ()                                                                                                                                              | Get the inverse parameters of the GLLiM model.                                                                         |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :ref:`directDensities <direct-densities-method>` (*x*, *x_incertitude=0*)                                                                                                              | Compute the direct densities given input matrix `x` and its uncertainties.                                             |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :ref:`inverseDensities <inverse-densities-method>` (*y*, *y_incertitude=0*)                                                                                                            | Compute the inverse densities given input matrix `y` and its uncertainties.                                            |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :ref:`getInsights <get-insights-method>` ()                                                                                                                                            | Returns ann Insights structure with informations about initialisation and training time, log-likelihood and arguments. |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+


    :ref:`Getters <getters>`
    ------------------------

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


    :ref:`Setters <setters>`
    ------------------------

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


    :ref:`Structures <structures>`
    ------------------------------

    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`GLLiMParameters <gllim-parameters-struct>`                | Describes the parameters of the GLLiM model **theta** = {**Pi**, **A**, **B**, **C**, **Gamma**, **Sigma**}. |
    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`GLLiMConstraints <gllim-constraints-struct>`              | Describes the constraints of the covariance matrices *Gamma* and *Sigma*.                                    |
    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`PredictionResult <prediction-result-struct>`              | Describes the results concerning a GLLiM density estimation (direct or inverse).                             |
    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`FullGMMResult <mean-prediction-result-struct>`     | Describes the results concerning a GLLiM density estimation by the mean.                                     |
    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`MergedGMMResult <center-prediction-result-struct>` | Describes the results concerning a GLLiM density estimation by the centroids.                                |
    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`Insights <insights-struct>`                               | Describes valuable information about initialisation and training (time, log-likelihood and configuration).   |
    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`InitialisationInsights <initialisation-insights-struct>`  | Describes valuable information about initialisation.                                                         |
    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`TrainingInsights <training-insights-struct>`              | Describes valuable information about training.                                                               |
    +-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
