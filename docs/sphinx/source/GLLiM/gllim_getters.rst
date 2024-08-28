.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>


This page describes the :ref:`GLLiM <gllim-class>` methods implying getting information on GLLiM's dimensions, constraints and :ref:`GLLiMParameters <gllim-parameters-structure>`.


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
