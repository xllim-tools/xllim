.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">Functional</div>


This page describes the :ref:`FunctionalModel <functional-model-class>` methods.

.. _functional-methods:

Methods
*******

.. _f-method:

.. method:: F(x, y)

    Calculate y = F(x) using armadillo library and write results to y without allocating new memory. This method is used only by the other components of the kernel.

    :param ndarray x: Vector of the functional model parameters (L dimension).
    :param ndarray y: Vector of results (D dimension).


.. _get-dimension-y-method:

.. method:: getDimensionY()

    Return the D dimension of the problem.

    :returns: The dimension D of the problem.


.. _get-dimension-x-method:

.. method:: getDimensionX()

    Return the L dimension of the problem.

    :returns: The dimension L of the problem.


.. _to-physic-method:

.. method:: toPhysic(x)

    Transform the values of x from the mathematical space to the physical space.

    :param ndarray x: The vector to normalize.


.. _from-physic-method:

.. method:: fromPhysic(x)

    Transform the values of x from the physical space to the mathematical space.

    :param ndarray x: The vector to normalize.


.. _gen-data-method:

.. method:: genData(N, generator_type, noise, seed)

    Generate a complete learning dataset from the generator type and the FunctionalModel.

    :param int N: Number of generated observations.
    :param str generator_type: The type of the generator used to generate x_gen matrix values.
    :param float, ndarray noise: Vector of dimension D corresponding to the y_i variances.
    :param int seed: Seed number for random generators.
    :returns: A generated dataset composed of a pair (x_gen, y_gen) with x_gen of shape (N, L) and y_gen of shape (N, D).


.. _importance-sampling-method:

.. method:: importanceSampling(proposition_gmms, y, y_err, N_0, B=0, J=0, covariance=0, idx_gaussian=-1, verbose=1, seed=0)

    Perform importance sampling with given parameters.

    :param proposition_gmms:
        List of GMM propositions. The GMMs can be defined by the three following objects :

        - (**list[(1-D ndarray, 2-D ndarray, 3-D ndarray)]**) A Python list with length N_obs containing each GMM defined as a tuple of 3 elements:

            - weigths (ndarray of shape (K)),
            - means (ndarray of shape (L, K)),
            - covariance matrices (ndarray of shape (K, L, L)).

        - (:class:`FullGMMResult`) The full GMM calculated with inverseDensities method.
        - (:class:`MergedGMMResult`) The merged GMM calculated with inverseDensities method.

    :type proposition_gmms: list[(1-D ndarray, 2-D ndarray, 3-D ndarray)], :class:`FullGMMResult`, :class:`MergedGMMResult`
        
    :param ndarray with shape(N_obs, D) y: Matrix y.
    :param ndarray with shape(N_obs, D) y_err: Matrix of y errors.
    
    :param int N_0: Initial number of samples.
    :param int B: (optional) Parameter B.
    :param int J: (optional) Parameter J.
    :param ndarray with shape(D) covariance: (optional) Covariance vector with shape (D).
    :param int idx_gaussian: (optional) Index of the desired gaussian from the merged GMM. Starts from 0 and ends at K_merged - 1. Perform importance sampling with given parameters on the specified gaussian of the GMMs.
    :param int verbose: (optional) The verbosity of the logging among {0, 1, 2}.
    :param int seed: (optional) The seed for random generation. It helps with reproducibility.

    :returns: An instance of :class:`ImportanceSamplingResult` containing the importance sampling results.
