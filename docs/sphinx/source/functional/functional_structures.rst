.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">Functional</div>


This page describes the :ref:`FunctionalModel <functional-model-class>` structures.



.. _functional-structures:

Structures
----------


.. _importance-sampling-result-struct:

.. class:: ImportanceSamplingResult(predictions, predictions_variance, nb_effective_sample, effective_sample_size, qn)

    A structure representing the result of the importance sampling algorithm.

    .. attribute:: predictions
        :type: ndarray of shape (L, N_obs)

        An array of shape (L, N_obs) containing the predicted values resulting from the importance sampling process.

    .. attribute:: predictions_variance
        :type: ndarray of shape (L, N_obs)

        An array of shape (L, N_obs) containing the variance of the predictions.

    .. attribute:: nb_effective_sample
        :type: ndarray of shape (N_obs)

        An array of shape (N_obs) of effective samples used during the importance sampling process.

    .. attribute:: effective_sample_size
        :type: ndarray of shape (N_obs)

        An array of shape (N_obs) of effective sample sizes, which is a measure of the quality of the importance sampling weights.
        Higher values indicate better quality of importance sampling.

    .. attribute:: qn
        :type: ndarray of shape (N_obs)

        An array of shape (N_obs) containing the normalized importance sampling weights.
