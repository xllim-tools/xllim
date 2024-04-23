#ifndef GLLIM_HPP
#define GLLIM_HPP

class GLLiM
{
public:
    /**
     * TODO
     */
    GLLiM(unsigned D, unsigned L, unsigned K, GLLiMParameters &theta, GLLiMConstraints &constraints);
    void initialize(
        const mat &x,
        const mat &y,
        unsigned seed,
        unsigned nb_iter_EM = 1,     // default = FixedInit
        unsigned nb_experiences = 1, // default = FixedInit
        // EMLearningConfig
        unsigned max_iteration,
        double ratio_ll,
        double floor,
        // GMMLearningConfig
        unsigned kmeans_iteration,
        unsigned em_iteration,
        double floor);
    void train(
        const mat &x,
        const mat &y,
        unsigned max_iteration,
        double ratio_ll,
        double floor);
    GLLiMParameters getParams();
    GLLiMParameters getParamA(); // one method for each GLLiM parameter
    void setParams(GLLiMParameters &theta);
    void setParamA(cube A); // one method for each GLLiM parameter
    GLLiMParameters getInverse();
    void directDensities(const mat &x);
    void inverseDensities(const mat &y);
    /// @brief TODO
    void getInsights();

private:
    // TODO
    std::shared_ptr<Iinitilizer<T, U>> initializer;                                            /**< @see Iinitilizer Iinitilizer*/
    std::shared_ptr<Iestimator<T, U>> estimator;                                               /**< @see Iestimator Iestimator*/
    std::shared_ptr<GLLiMParameters<T, U>> gllim_parameters;                                   /**< The parameters of the direct GLLiM model*/
    std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> inverse_gllim_parameters; /**< The parameters of the inverse GLLiM model*/
    unsigned K;                                                                                /**< the number of affine transformation and the number of gaussian distributions in the mixture */

protected:
    // TODO
    /**
     * This method adjusts the Sigma parameter of the trained GLLiM model with the variance of the observation before computing the corresponding GMM
     * @param gllim : the parameters of the trained GLLiM
     * @param cov_obs : the variance or measure error of the observation
     */
    void alterCovariance(GLLiMParameters<T, U> &gllim, const vec &cov_obs);
};

#endif // GLLIM_HPP
