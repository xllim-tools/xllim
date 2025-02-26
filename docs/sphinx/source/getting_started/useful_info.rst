.. _useful_info:

Workflow summary
----------------

This is a very concise summary of the general workflow.

- Without functional model, from a dataset:

    .. code-block:: python

        gllim = xllim.GLLiM(K, D, L, gamma_type="full", sigma_type="diag", n_hidden_variables=n_hidden_variables)
        gllim.initialize(x_gen, y_gen, *initialisation args*)
        gllim.train(x_gen, y_gen, *training args*)
        predictions = gllim.inverseDensities(y_obs, y_obs_noise)


- With functional model:

    .. code-block:: python

        model = xllim.TestModel()
        x_gen, y_gen = model.genData(N, *args*)
        gllim = xllim.GLLiM(K, D, L, gamma_type="full", sigma_type="diag", n_hidden_variables=n_hidden_variables)
        gllim.initialize(x_gen, y_gen, *initialisation args*)
        gllim.train(x_gen, y_gen, *training args*)
        predictions = gllim.inverseDensities(y_obs, y_obs_noise)
        is_results = model.importanceSampling(predictions.fullGMM, y_obs, y_obs_noise, N_0, *kwargs*)



Useful dimension parameters
----------------------------

+--------+------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Symbol |             Parameter              |                                                                                                                                          Description                                                                                                                                           |
+========+====================================+================================================================================================================================================================================================================================================================================================+
| K      | Number of Gaussians in GLLiM model | It corresponds to the number of affine transformations in the GLLiM model.                                                                                                                                                                                                                     |
+--------+------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| D      | Output (Y) dimension               | The dimension of the model output corresponds to the *Y* vector dimension such that the forward model is represented by *Y=F(X)*.                                                                                                                                                              |
+--------+------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| L      | Input (X) dimension                | The dimension of the model output corresponds to the *X* vector dimension such that the forward model is represented by *Y=F(X)*. It also represents the number of features composed of observed and latent variables such that **L = L_t + L_w**.                                             |
+--------+------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| L_t    | Observed input dimension           | It corresponds to the number of observed features.                                                                                                                                                                                                                                             |
+--------+------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| L_w    | Latent input dimension             | It corresponds to the number of unobserved/hidden features.                                                                                                                                                                                                                                    |
|        |                                    |                                                                                                                                                                                                                                                                                                |
|        |                                    | - Lw = 0. This is the fully supervised case, and is equivalent to the mixture of local linear experts (MLE) model.                                                                                                                                                                             |
|        |                                    | - Lw = D. Σ takes the form of a general covariance matrix and we obtain the JGMM model. This is the most general GLLiM model, which requires the estimation of K full covariance matrices of size (D + L) × (D + L). This model becomes over-parameterized and intractable in high dimensions. |
|        |                                    | - 0 < Lw < D. This corresponds to the hybrid GLLiM model, and yields a wide variety of novel regression models in between MLE and JGMM.                                                                                                                                                        |
+--------+------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| N      | Dataset size                       | It is the number of pair (X,Y) in the dataset used to train the GLLiM model.                                                                                                                                                                                                                   |
+--------+------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| N_obs  | Number of observation              | It is related to the number of observations, usually the model output *Y*. It refers to the number of observed vector we want to apply the trained GLLiM model on.                                                                                                                             |
+--------+------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Shape of all ndarrays
---------------------

⚠️ Note that all ndarrays must be filled with double-precision floating-point, ie Numpy.ndarray[numpy.float64]

+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Name  |   Shape   |                                                                                 Note                                                                                  |
+=======+===========+=======================================================================================================================================================================+
| Pi    | (K,)      | :class:`GLLiMParameters.Pi`                                                                                                                                           |
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| A     | (K, D, L) | :class:`GLLiMParameters.A`                                                                                                                                            |
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| B     | (K, D)    | :class:`GLLiMParameters.B`                                                                                                                                            |
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| C     | (K, L)    | :class:`GLLiMParameters.C`                                                                                                                                            |
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gamma | (K, L, L) | According to the covariance matrix type the shape can be adapted in order to not have useless zeros. ('full',)                                                        |
|       |           |                                                                                                                                                                       |
|       |           | - In the case of Full covariance matrix (*gamma_type = 'full'*), Gamma is of shape (K, L, L).                                                                         |
|       |           | - In the case of Diagonal covariance matrix (*gamma_type = 'diag'*), Gamma is of shape (K, L) with Gamma[k] representing the variances vector of the k^{th} gaussian. |
|       |           | - In the case of Isotropic covariance matrix (*gamma_type = 'iso'*), Gamma is of shape (K) with Gamma[k] representing the unique variance of the k^{th} gaussian.     |
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Sigma | (K, D, D) | Idem to Gamma.                                                                                                                                                        |
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| X     | (L, N)    | It applies to X_gen, X_obs and X_pred with the corresponding N_gen, N_obs dimension.                                                                                  |
|       |           | X_pred can be estimated by PredictionResult.mergedGMM.mean or ImportanceSamplingResult.predictions                                                                    |
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Y     | (D, N)    | Idem to X.                                                                                                                                                            |
+-------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
