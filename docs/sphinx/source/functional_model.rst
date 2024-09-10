FunctionalModel Class
======================

The base class :class:`FunctionalModel` is an abstract class representing the functional model.
It offers the functional method "F" which requires that the parameters of X be
in mathematical space. It contains normalization methods to transform
X from and to physical space. It also allows to retrieve the dimensions
of the problem.


:ref:`Derived classes<derived-classes>`
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


:ref:`Methods<methods>`
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

.. _methods:

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

.. method:: importanceSampling(proposition_gmms, y, y_err, covariance, N_0, B=0, J=0)

    Perform importance sampling with given parameters.

    :param list[1-D ndarray, 2-D ndarray, 3-D ndarray] proposition_gmms: List of GMM propositions.
    :param mat y: Matrix y.
    :param mat y_err: Matrix of y errors.
    :param ndarray covariance: Covariance vector.
    :param int N_0: Initial number of samples.
    :param int B: (optional) Parameter B.
    :param int J: (optional) Parameter J.
    :returns: An instance of `ImportanceSamplingResult` containing the importance sampling results.


.. _derived-classes:

Derived classes
***************

.. _hapke-model:

Hapke Model
-----------

.. class:: HapkeModel (geometries, variant, adapter, theta_bar_scaling, b0, h)

    The class :class:`HapkeModel` is a representation of the Hapke's photometric model.
    For more details check the scientific :ref:`documentation <DocHapkeModel>`.

    There are two variants of the Hapke's formulation, the initial 1993 version and the 2002 version.
    The calculation in the class are initially designed for a 4 parameters model. The class can
    use a adapter to adapt different variants of the model for example : the model with 3 or 6 parameters.

    See : Hapke B. 1993 Theory of Reflectance and Emittance Spectroscopy. Topics in Remote Sensing.
    Cambridge University Press, Cambridge, UK.

    See : Schmidt F. and Fernando J. 2015 Realistic uncertainties on Hapke model parameters from
    photometric measurement. Icarus, 260 :73 - 93, 2015.

    :param ndarray of shape (n_geometries,3) geometries: The matrix of geometries that will be used by the model. The shape of the Numpy array should be (n_geometries,3).
        The three geometric angles must be this particular order :
            - At [:,0] the incidence angle (*inc*) also called the solar zenith angle (*sza*)
            - At [:,1] the emergence angle (*eme*) also called the vertical zenith angle (*vza*)
            - At [:,2] the phi angle (*phi*)
        Thus the geometries matrix should look like this

        >>> geometries
        array([ [inc_0, eme_0, phi_0],
                [inc_1, eme_1, phi_1],
                [inc_2, eme_2, phi_2],
                        ...          
                [inc_n, eme_n, phi_n]])                

    :param string variant: The variant a the Hapke's model among {'1993', '2002'}.
    :param string adapter: This argument allows to adapt the Hapke's model parameter number among {'three', 'four', 'six'}.
    :param float theta_bar_scaling: Value used to transform theta_bar between physical and mathematical spaces.
    :param float b0: The amplitude of the opposition effect
    :param float h: The angular width of the opposition effect


.. _shkuratov-model:

Shkuratov Model
---------------

.. class:: ShkuratovModel (geometries, variant, scaling_coeffs, offset)

    The class :class:`ShkuratovModel` is a representation of the Shkuratov's photometric model.
    For more details check the scientific :ref:`documentation <DocShkuratovModel>`.

    There are two variants of the Shkuratov's formulation, the original model with 5 parameters and a reduced model with 3 parameters.
    The *scaling_coeffs* and *offset* arguments are  to perform affine transformation between the physical space and the mathematical space, such as, 
    :math:`(\text{toPhysic}(x))_{1 \leq i \leq L} = (\text{scaling_coeffs}_i x_i + \text{offset}_i)_{1 \leq i \leq L}`.

    :param ndarray of shape (n_geometries,3) geometries: The matrix of geometries that will be used by the model. The shape of the Numpy array should be (n_geometries,3).
        The three geometric angles must be this particular order :
            - At [:,0] the incidence angle (*inc*) also called the solar zenith angle (*sza*)
            - At [:,1] the emergence angle (*eme*) also called the vertical zenith angle (*vza*)
            - At [:,2] the phi angle (*phi*)
        Thus the geometries matrix should look like this

        >>> geometries
        array([ [inc_0, eme_0, phi_0],
                [inc_1, eme_1, phi_1],
                [inc_2, eme_2, phi_2],
                        ...          
                [inc_n, eme_n, phi_n]])                

    :param string variant: The variant a the Hapke's model among {'3p', '5p'}. '3p' refers to the 3-parameters variant- and '5p' refers to the original 5-parameters model.
    :param ndarray of shape (L,) scaling_coeffs: A set of coefficients used in the transformation between physical and mathematical spaces.
    :param ndarray of shape (L,) offset: A set of offsets used in the transformation between physical and mathematical spaces


.. _external-python-model:

External Python Model
---------------------

.. class:: ExternalPythonModel (className, fileName, filePath)

    The PlanetGLLiM software allows you to edit your own physical model. To do so you must provide a Python file (.py) 
    describing your physical model as a Python class. The template of such python module is described below.

    >>> # IMPORTS
    >>> import numpy as np
    >>>
    >>> class PhysicalModel(object):
    >>>    """ This is a python class defining a functional model. 
    >>>    
    >>>    This class is composed of 5 mandatory functions:
    >>>        - F: the functional model F describing the physical model. F takes photometries as arguments and return reflectances
    >>>        - getDimensionY: returns the dimension of Y (reflectances)
    >>>        - getDimensionX: return de dimension of X (photometries)
    >>>        - toPhysic: converts the X data from mathematical framework (0<X<1) to physical framework
    >>>        - fromPhysic: converts the X data from physical framework to mathematical framework (0<X<1)
    >>>
    >>>    Note that some class constants, other functions and class constructors can be declared.
    >>>
    >>>    Geometries : if your physical model requires geometries as with Shkuratov, the structure and the values of the geometries
    >>>        must be declared within this Python file.
    >>>
    >>>    See the Planet-Gllim documentation for more informations (https://gitlab.inria.fr/kernelo-mistis/planet-gllim-front-end/-/wikis/home)
    >>>    """
    >>>    
    >>>    #################################################################################################
    >>>    ##                          CLASS CONSTANTS (OPTIONAL)                                         ##
    >>>    #################################################################################################
    >>>
    >>>    # Geometries of the physical model MUST be declared here - directly in the python file. 
    >>>
    >>>
    >>>    #################################################################################################
    >>>    ##                          CORE FUNCTIONS (MANDATORY)                                         ##
    >>>    #################################################################################################
    >>>
    >>>    def F(self):
    >>>        pass
    >>>
    >>>    def getDimensionY(self):
    >>>        pass
    >>>
    >>>    def getDimensionX(self):
    >>>        pass
    >>>
    >>>    def toPhysic(self):
    >>>        pass
    >>>
    >>>    def fromPhysic(self):
    >>>        pass
    >>>
    >>>
    >>>    #################################################################################################
    >>>    ##                          OTHER FUNCTIONS (OPTIONAL)                                         ##
    >>>    #################################################################################################
    >>>
    >>>    def __init__(self, *args, **kwargs):
    >>>        pass
    >>>
    >>>    def optional_but_useful_function():
    >>>        pass

    If the functional model shows interest it can be added in further development in Kernelo as a built-in class to 
    improve speed performance.

    :param string className: The exact name of the python class.
    :param string fileName: The exact name of the python module without the .py extension.
    :param string filePath: The directory path of the python module.


.. _test-model:

Test Model
----------

.. class:: TestModel ()

    The class :class:`TestModel` is low-dimensional functional model implemented in order to test the GLLiM method with simple but not trivial example.
    The functional F is designed so as to exhibit 2 solutions with D=9 and L=4. :math:`F = A ◦ G ◦ H`, where
        - :math:`A` is a (DxL) injective matrix,
    
        .. math::
            A = \frac{1}{2} \begin{pmatrix}
                1 & 2 & 2 & 1 \\
                0 & 0.5 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 3 \\
                0.2 & 0 & 0 & 0 \\
                0 & -0.5 & 0 & 0 \\
                -0.2 & 0 & -1 & 0 \\
                -1 & 0 & 2 & 0 \\
                0 & 0 & 0 & -0.7
            \end{pmatrix}

        - :math:`G(x)` = :math:`(exp(x_1), exp(x_2), exp(x_3), exp(x_4))` and 
        - :math:`H(x)` = :math:`(x_1, x_2, 4(x_3-0.5)^2, x_4)`.
    
    The resulting F is therefore non-linear and yields two solutions for each observation, denoted by xobs,1 and xobs,2 = 1 - xobs,1.
