.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>



.. _gllim-class:

GLLiM class overall
===================

.. class:: GLLiM (L, D, K, gamma_type, sigma_type)

    Gaussian Locally-Linear Model (GLLiM) for probabilistic modeling.

    :param int L: The latent space dimension.
    :param int D: The observed space dimension.
    :param int K: The number of Gaussian components.
    :param str gamma_type: The type of gamma parameter among {*'full'*, *'diag'*, *'iso'*}.
    :param str sigma_type: The type of sigma parameter among {*'full'*, *'diag'*, *'iso'*}.
    :returns: An instance of the GLLiM class.


    .. toctree::
        :hidden:

        gllim_main_methods
        gllim_getters
        gllim_setters
        gllim_structures

            
    :ref:`Main methods <gllim-main-methods>`
    ----------------------------------------

    +----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :ref:`initialize<initialize-method>`               | Initialize the GLLiM model with given data and parameters.                                                                                           |
    +----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :ref:`train <train-method>`                        | Train the GLLiM model with given data and parameters.                                                                                                |
    +----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :ref:`trainJGMM <train-jgmm-method>`               | Train the Joint GLLiM model using Armadillo built-in EM algorithm. This method is only available with (*gamma_type* = 'full', *sigma_type* = 'full') |
    +----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :ref:`getInverse <get-inverse-method>`             | Get the inverse parameters of the GLLiM model.                                                                                                       |
    +----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :ref:`directDensities <direct-densities-method>`   | Compute the direct densities given input matrix `x` and its uncertainties.                                                                           |
    +----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :ref:`inverseDensities <inverse-densities-method>` | Compute the inverse densities given input matrix `y` and its uncertainties.                                                                          |
    +----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :ref:`getInsights <get-insights-method>`           | Returns ann Insights structure with informations about initialisation and training time, log-likelihood and arguments.                               |
    +----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------+


    :ref:`Getters <gllim-getters>`
    ------------------------------

    +------------------------------------------------+-----------------------------------------+
    | :ref:`getDimensions <get-dimensions-method>`   | Get the dimensions of the GLLiM model.  |
    +------------------------------------------------+-----------------------------------------+
    | :ref:`getConstraints <get-constraints-method>` | Get the constraints of the GLLiM model. |
    +------------------------------------------------+-----------------------------------------+
    | :ref:`getParams <get-params-method>`           | Get the parameters of the GLLiM model.  |
    +------------------------------------------------+-----------------------------------------+
    | :ref:`getParamPi <get-param-pi-method>`        | Get the mixture coefficients `Pi`.      |
    +------------------------------------------------+-----------------------------------------+
    | :ref:`getParamA <get-param-a-method>`          | Get the parameter matrix `A`.           |
    +------------------------------------------------+-----------------------------------------+
    | :ref:`getParamB <get-param-b-method>`          | Get the parameter matrix `B`.           |
    +------------------------------------------------+-----------------------------------------+
    | :ref:`getParamC <get-param-c-method>`          | Get the parameter matrix `C`.           |
    +------------------------------------------------+-----------------------------------------+
    | :ref:`getParamGamma <get-param-gamma-method>`  | Get the gamma parameters.               |
    +------------------------------------------------+-----------------------------------------+
    | :ref:`getParamSigma <get-param-sigma-method>`  | Get the sigma parameters.               |
    +------------------------------------------------+-----------------------------------------+


    :ref:`Setters <gllim-setters>`
    ------------------------------

    +-----------------------------------------------+----------------------------------------+
    | :ref:`setParams <set-params-method>`          | Set the parameters of the GLLiM model. |
    +-----------------------------------------------+----------------------------------------+
    | :ref:`setParamPi <set-param-pi-method>`       | Set the mixture coefficients `Pi`.     |
    +-----------------------------------------------+----------------------------------------+
    | :ref:`setParamA <set-param-a-method>`         | Set the parameter matrix `A`.          |
    +-----------------------------------------------+----------------------------------------+
    | :ref:`setParamB <set-param-b-method>`         | Set the parameter matrix `B`.          |
    +-----------------------------------------------+----------------------------------------+
    | :ref:`setParamC <set-param-c-method>`         | Set the parameter matrix `C`.          |
    +-----------------------------------------------+----------------------------------------+
    | :ref:`setParamGamma <set-param-gamma-method>` | Set the gamma parameters.              |
    +-----------------------------------------------+----------------------------------------+
    | :ref:`setParamSigma <set-param-sigma-method>` | Set the sigma parameters.              |
    +-----------------------------------------------+----------------------------------------+


    :ref:`Structures <gllim-structures>`
    ------------------------------------

    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`GLLiMParameters <gllim-parameters-struct>`               | Describes the parameters of the GLLiM model **theta** = {**Pi**, **A**, **B**, **C**, **Gamma**, **Sigma**}. |
    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`GLLiMConstraints <gllim-constraints-struct>`             | Describes the constraints of the covariance matrices *Gamma* and *Sigma*.                                    |
    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`PredictionResult <prediction-result-struct>`             | Describes the results concerning a GLLiM density estimation (direct or inverse).                             |
    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`FullGMMResult <full-gmm-result-struct>`                  | Describes the results concerning a GLLiM density estimation by the mean.                                     |
    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`MergedGMMResult <merged-gmm-result-struct>`              | Describes the results concerning a GLLiM density estimation by the centroids.                                |
    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`Insights <insights-struct>`                              | Describes valuable information about initialisation and training (time, log-likelihood and configuration).   |
    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`InitialisationInsights <initialisation-insights-struct>` | Describes valuable information about initialisation.                                                         |
    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
    | :ref:`TrainingInsights <training-insights-struct>`             | Describes valuable information about training.                                                               |
    +----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
