.. raw:: html

   <div style="text-align: center; font-size: 36px; font-weight: bold;">Functional</div>



.. _functional-model-class:

FunctionalModel class overall
=============================

.. class:: FunctionalModel

    The base class :class:`FunctionalModel` is an abstract class representing the functional model.
    It offers the functional method "F" which requires that the parameters of X be
    in mathematical space. It contains normalization methods to transform
    X from and to physical space. It also allows to retrieve the dimensions of the problem.


    .. toctree::
        :hidden:

        functional_derived_classes
        functional_methods
        functional_structures


    :ref:`Derived classes<functional-derived-classes>`
    ---------------------------------------

    +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
    | :ref:`HapkeModel <hapke-model>`                    | The ``HapkeModel`` class describes the Hapke photometric model.                                               |
    +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
    | :ref:`ShkuratovModel <shkuratov-model>`            | The ``ShkuratovModel`` class describes the Shkuratov photometric model.                                       |
    +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
    | :ref:`ExternalPythonModel <external-python-model>` | The ``ExternalPythonModel`` class allows to import a python script in order to use your own functional model. |
    +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
    | :ref:`TestModel <test-model>`                      | The ``TestModel`` class describes a simple non-linear model                                                   |
    +----------------------------------------------------+---------------------------------------------------------------------------------------------------------------+


    :ref:`Methods<functional-methods>`
    -----------------------

    +------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | :ref:`F <f-method>` (*x*)                                                                                                    | Apply the model function on vector *x*                                     |
    +------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | :ref:`getDimensionY <get-dimension-y-method>` ()                                                                             | Get the dimension **D** of the model - ie. dim(*Y*)                        |
    +------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | :ref:`getDimensionX <get-dimension-x-method>` ()                                                                             | Get the dimension **L** of the model - ie. dim(*X*)                        |
    +------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | :ref:`toPhysic <to-physic-method>` (*x*)                                                                                     | Transform the values of x from the mathematical space to the physical.     |
    +------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | :ref:`fromPhysic <from-physic-method>` (*x*)                                                                                 | Transform the values of x from the physical space to the mathematical.     |
    +------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | :ref:`genData <gen-data-method>` (*N*, *generator_type*, *covariance*, *seed*)                                               | Generate a complete learning dataset with given covariance or noise ratio. |
    +------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | :ref:`importanceSampling <importance-sampling-method>` (*proposition_gmms*, *y*, *y_err*, *covariance*, *N_0*, *B=0*, *J=0*) | Perform importance sampling with given parameters.                         |
    +------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+


    :ref:`Structures <functional-structures>`
    ------------------------------

    +---------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
    | :ref:`importanceSamplingResult <importance-sampling-result-struct>` | Describes the results concerning the :ref:`importanceSampling <importance-sampling-method>` method. |
    +---------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
