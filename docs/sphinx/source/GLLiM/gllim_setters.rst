.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">GLLiM</div>


This page describes the :ref:`GLLiM <gllim-class>` methods implying setting :ref:`GLLiMParameters <gllim-parameters-structure>`.


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
