.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>


This page describes the :ref:`GLLiM <gllim-class>` methods implying setting :ref:`GLLiMParameters <gllim-parameters-struct>`.


.. _gllim-setters:

Setters
-------

.. _set-params-method:

    .. method:: setParams(theta)

        Set the parameters of the GLLiM model.

        :param GLLiMParameters theta:


.. _set-param-pi-method:

    .. method:: setParamPi(Pi)

        Set the mixture coefficients `Pi`.

        :param ndarray of shape (K) Pi:


.. _set-param-a-method:

    .. method:: setParamA(A)

        Set the parameter matrix `A`.

        :param ndarray of shape (D, L, K) A:


.. _set-param-b-method:

    .. method:: setParamB(B)

        Set the parameter matrix `B`.

        :param ndarray of shape (D, K) B:


.. _set-param-c-method:

    .. method:: setParamC(C)

        Set the parameter matrix `C`.

        :param ndarray of shape (L, K) C:


.. _set-param-gamma-method:

    .. method:: setParamGamma(Gamma)

        Set the gamma parameters. Shape depends on Gamma constraints.
        Gamma is a ndarray containing the K covariance matrices of the mixture of Gaussian distributions that define the low-dimensional data.

        - In the case of Full covariance matrix (*gamma_type = 'full'*), Gamma is of shape (K, L, L).
        - In the case of Diagonal covariance matrix (*gamma_type = 'diag'*), Gamma is of shape (K, L) with Gamma[k] representing the variances vector of the k^{th} gaussian.
        - In the case of Isotropic covariance matrix (*gamma_type = 'iso'*), Gamma is of shape (K) with Gamma[k] representing the unique variance of the k^{th} gaussian.

        :param ndarray of shape (K, L*, L*) Gamma:


.. _set-param-sigma-method:

    .. method:: setParamSigma(Sigma)

        Set the sigma parameters. Shape depends on Gamma constraints.
        Sigma is a ndarray containing the K covariance matrices of the mixture of Gaussian distributions that define the high-dimensional data.

        - In the case of Full covariance matrix (*gamma_type = 'full'*), Sigma is of shape (K, D, D).
        - In the case of Diagonal covariance matrix (*gamma_type = 'diag'*), Sigma is of shape (K, D) with Sigma[k] representing the variances vector of the k^{th} gaussian.
        - In the case of Isotropic covariance matrix (*gamma_type = 'iso'*), Sigma is of shape (K) with Sigma[k] representing the unique variance of the k^{th} gaussian.

        :param ndarray of shape (K, D*, D*) Sigma:
