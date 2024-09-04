#ifndef FUNCTIONALMODEL_HPP
#define FUNCTIONALMODEL_HPP

#include <armadillo>

using namespace arma;

struct ImportanceSamplingResult
{
    mat predictions;
    mat predictions_variance;
    rowvec nb_effective_sample;
    rowvec effective_sample_size;
    rowvec qn;

    ImportanceSamplingResult(unsigned L, unsigned N_obs) : predictions(L, N_obs), predictions_variance(L, N_obs), nb_effective_sample(N_obs), effective_sample_size(N_obs), qn(N_obs) {}
};

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

    /** @brief This method generates a complete learning data set from the generator type and the @class Functional model.
     * @param N : number of generated observation
     * @param generator_type : the type of the generator used to generate x_gen matrix values
     * @param covariance : vector of dimension D coresponding to the y_i variances.
     * @param seed : seed number for random generators
     * @return A generated dataset composed of a pair (x_gen, y_gen) with x_gen of shape (N,L) and y_gen of shape (N,D)
     */
    std::tuple<mat, mat> genData(unsigned N, const std::string &generator_type, vec &covariance, unsigned seed);

    /** @brief This method generates a complete learning data set from the generator type and the @class Functional model.
     * @param N : number of generated observation
     * @param generator_type : the type of the generator used to generate x_gen matrix values
     * @param noise_ratio : noise effect
     * @param seed : seed number for random generators
     * @return A generated dataset composed of a pair (x_gen, y_gen) with x_gen of shape (N,L) and y_gen of shape (N,D)
     */
    std::tuple<mat, mat> genData(unsigned N, const std::string &generator_type, double noise_ratio, unsigned seed = 0);

    /**
    args: 
        gmm_weights (K)
        gmm_means (L,K)
        gmm_covs (L,LK)
        y (N,L)
    return:
        results.predictions (L,N)
     */
    ImportanceSamplingResult importanceSampling(std::vector<std::tuple<const vec, const mat, const cube>> proposition_gmms, const mat y, const mat y_err, const vec covariance, const unsigned N_0, const unsigned B = 0, const unsigned J = 0, int verbose = 1);
    // NOTE: si on met "const vec/mat/cube &" on obtient une erreur avec CARMA/python "Memoryerror: std::bad_alloc"

protected:
    /**
     * TODO
     */
    vec targetDensity(const mat &x, const vec &y, const vec &y_err, const vec &covariance, bool log = true);

    /**
     * TODO
     */
    void targetDensity(vec &x, vec &y, vec &y_err, double noise_ratio, bool log = true);

    /**
     * TODO
     */
    vec propositionDensity(const mat &x, const vec &weight, const mat &mean, const cube &covariance, bool log = true);

    /**
     * TODO
     */
    // void propositionDensity(vec &x, vec &y, vec &y_err, vec &noise, bool log = true);
};

#endif // FUNCTIONALMODEL_HPP