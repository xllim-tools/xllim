.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>


This page describes the :ref:`GLLiM <gllim-class>` methods implying getting information on GLLiM's dimensions, constraints and :ref:`GLLiMParameters <gllim-parameters-struct>`.


.. _getters:

Getters
-------

.. _get-dimensions-method:

    .. method:: getDimensions()

        Get the dimensions of the GLLiM model.

        :returns:
            (*string*) A string describing the dimensions of the model.


.. _get-constraints-method:

    .. method:: getConstraints()

        Get the constraints of the GLLiM model.

        :returns:
            (*string*): A string describing the constraints of the model.


.. _get-params-method:

    .. method:: getParams()

        Get the parameters of the GLLiM model.

        :returns:
            (*GLLiMParameters*): An instance of :ref:`GLLiMParameters <gllim-parameters-struct>` containing the model parameters.


.. _get-param-pi-method:

    .. method:: getParamPi()

        Get the mixture coefficients `Pi`.

        :returns: 
            (*ndarray of shape (K)*): A row vector of mixture coefficients.


.. _get-param-a-method:

    .. method:: getParamA()

        Get the parameter matrix `A`.

        :returns:
            (*ndarray of shape (D, L, K)*): A cube containing the parameter matrix `A`.


.. _get-param-b-method:

    .. method:: getParamB()

        Get the parameter matrix `B`.

        :returns:
            (*ndarray of shape (D, K)*): A matrix containing the parameter matrix `B`.


.. _get-param-c-method:

    .. method:: getParamC()

        Get the parameter matrix `C`.

        :returns:
            (*ndarray of shape (L, K)*): A matrix containing the parameter matrix `C`.


.. _get-param-gamma-method:

    .. method:: getParamGamma()

        Get the gamma parameters.

        :returns:
            (*ndarray of shape (K, L, L)*):
            Gamma is a ndarray containing the K covariance matrices of the mixture of Gaussian distributions that define the low-dimensional data.
                - In the case of Full covariance matrix (*gamma_type = 'full'*), Gamma is of shape (K, L, L).
                - In the case of Diagonal covariance matrix (*gamma_type = 'diag'*), Gamma is of shape (K, L) with Gamma[k] representing the variances vector of the k^{th} gaussian.
                - In the case of Isotropic covariance matrix (*gamma_type = 'iso'*), Gamma is of shape (K) with Gamma[k] representing the unique variance of the k^{th} gaussian.


.. _get-param-sigma-method:

    .. method:: getParamSigma()

        Get the sigma parameters.

        :returns: 
            (*ndarray of shape (K, D, D)*):
            Sigma is a ndarray containing the K covariance matrices of the mixture of Gaussian distributions that define the high-dimensional data.
                - In the case of Full covariance matrix (*gamma_type = 'full'*), Sigma is of shape (K, D, D).
                - In the case of Diagonal covariance matrix (*gamma_type = 'diag'*), Sigma is of shape (K, D) with Sigma[k] representing the variances vector of the k^{th} gaussian.
                - In the case of Isotropic covariance matrix (*gamma_type = 'iso'*), Sigma is of shape (K) with Sigma[k] representing the unique variance of the k^{th} gaussian.
