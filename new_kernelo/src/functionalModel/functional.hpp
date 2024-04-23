#ifndef FUNCTIONNAL_HPP
#define FUNCTIONNAL_HPP

/**
 * @class FunctionalModel
 * @brief Abstract class representing the functional model
 *
 * This class is an interface of the functional model. It offers the
 * functional method "F" which requires that the parameters of X be
 * in mathematical space. It contains normalization methods to transform
 * X from and to physical space. It also allows to retrieve the dimensions
 * of the problem.
 *
 */
class FunctionalModel
{
public:
    /**
     * This method calculates y = F(x) using armadillo library and writes
     * results on the vector y without allocating new memory. This method
     * is used only by the other components of the kernel.
     *
     * @param x : vector of the functional model parameters. (L dimension)
     * @param y : vector of results (D dimension)
     */
    virtual void F(vec x, vec &y) = 0;

    /**
     * This method returns the D dimension of the problem
     * @return the dimension D of the problem
     */
    virtual unsigned getDimensionY() = 0;

    /**
     * This method returns the L dimension of the problem
     * @return the dimension L of the problem
     */
    virtual unsigned getDimensionX() = 0;

    /**
     * This method transforms the values of x from the mathematical
     * space to the physical space.
     * @param x : the vector to normalize
     */
    virtual void toPhysic(vec &x) = 0;

    /**
     * This method transforms the values of x from the mathematical
     * space to the physical space.
     * @param x : the vector to normalize
     */
    virtual void fromPhysic(vec &x) = 0;

    /**
     * TODO
     */
    void genData(unsigned N, std::string &generator_type, vec &noise, unsigned seed = 0);

    /**
     * TODO
     */
    void genData(unsigned N, std::string &generator_type, double noise_ratio, unsigned seed = 0);

    /**
     * TODO
     */
    void importanceSampling(vec &weights, mat &means, cube &covariances, vec &y, vec &y_err, unsigned N_0, unsigned B = 0, unsigned J = 0);

protected:
    /**
     * TODO
     */
    void targetDensity(vec &x, vec &y, vec &y_err, vec &noise, bool log = true);

    /**
     * TODO
     */
    void targetDensity(vec &x, vec &y, vec &y_err, double noise_ratio, bool log = true);
};

#endif // FUNCTIONNAL_HPP
