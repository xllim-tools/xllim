#ifndef EMESTIMATOR_HPP
#define EMESTIMATOR_HPP

#include "gllim.hpp"

/**
 * @class EmEstimator
 * @brief Expectation-Maximization based estimator
 * @details This estimator uses the algorithm Expectation-Maximization to train the GLLiM model. The E-step is referred
 * by the method 'next_rnk' and the M-step by the method next_theta.
 * @tparam T : the type of Gamma matrices must be a specialisation of @see Icovariance Icovariance.
 * @tparam U : the type of Sigma matrices must be a specialisation of @see Icovariance Icovariance.
 */
class EmEstimator
{
public:
    EmEstimator();
    static GLLiMParameters train(const mat &x, const mat &y, GLLiMParameters &theta, unsigned max_iteration, double ratio_ll, double floor);

//     /**
//      * @brief E-step method
//      * @details The method computes the posterior as indicated in the formula 27 in : Antoine Deleforge, Florence Forbes,
//      * and Radu Horaud. High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables.
//      * Statistics and Computing 25(5): 893-911, September 2015. The logarithm of the posterior is returned for computation
//      * stability.
//      * @param x : a matrix of low dimension data
//      * @param y : a matrix of high dimension data
//      * @param theta : the current value of the parameters of the GLLiM model.
//      * @param next_rnk : the new posterior computed by the E-step.
//      */
//     void next_rnk(const mat &x, const mat &y, std::shared_ptr<GLLiMParameters<T, U>> theta, mat &next_rnk);

//     /**
//      * @brief M-step method
//      * @details The methods performs the update of the parameters of the GLLim Model using the new value of the posterior rnk.
//      * See page 900 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with Gaussian Mixtures and
//      * Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
//      * @param x : a matrix of low dimension data
//      * @param y : a matrix of high dimension data
//      * @param r_nk : the logarithm of the posterior
//      * @param next_theta : the new values of the GLLiM model parameters computed in this method
//      */
//     void next_theta(const mat &x, const mat &y, const mat &r_nk, std::shared_ptr<GLLiMParameters<T, U>> next_theta);

//     mat norm_log_rnk(const mat &r_nk);
//     double log_likelihood(const mat &r_nk);

// private:
//     std::shared_ptr<EMLearningConfig> config; /**< The estimator configuration parameters @see EMLearningConfig EMLearningConfig*/

//     /**
//      * @brief update of the parameter A of the GLLiM model
//      * @details See the formulas 31 to 36 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with
//      * Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
//      * @param next_theta : @see GLLiMParameters GLLiMParameters
//      * @param k : number of affine transformations
//      * @param x : a matrix of low dimension data
//      * @param y : a matrix of high dimension data
//      * @param exp_avg_rnk : the mean of the posteriors
//      */
//     void update_A_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const mat &y, const vec &exp_avg_rnk);

//     /**
//      * @brief update of the parameter B of the GLLiM model
//      * @details See the formula 37 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with
//      * Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
//      * @param next_theta : @see GLLiMParameters GLLiMParameters
//      * @param k : number of affine transformations
//      * @param Y_AX : is the result of the computation of y - A * x
//      * @param exp_avg_rnk : the mean of the posteriors
//      */
//     void update_B_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &Y_AX, const vec &exp_avg_rnk);

//     /**
//      * @brief update of the parameter Sigma of the GLLiM model
//      * @details See the formula 38 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with
//      * Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
//      * @param next_theta : @see GLLiMParameters GLLiMParameters
//      * @param k : number of affine transformations
//      * @param Y_AX : is the result of the computation of y - A * x
//      * @param exp_avg_rnk : the mean of the posteriors
//      */
//     void update_Sigma_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &Y_AX, const vec &exp_avg_rnk);

//     /**
//      * @brief update of the parameter Gamma of the GLLiM model
//      * @details See the formula 30 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with
//      * Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
//      * @param next_theta : @see GLLiMParameters GLLiMParameters
//      * @param k : number of affine transformations
//      * @param x : a matrix of low dimension data
//      * @param exp_avg_rnk : the mean of the posteriors
//      */
//     void update_Gamma_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const vec &exp_avg_rnk);

//     /**
//      * @brief update of the parameter C of the GLLiM model
//      * @details See the formula 29 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with
//      * Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
//      * @param next_theta : @see GLLiMParameters GLLiMParameters
//      * @param k : number of affine transformations
//      * @param x : a matrix of low dimension data
//      * @param exp_avg_rnk : the mean of the posteriors
//      */
//     void update_C_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const vec &exp_avg_rnk);

//     /**
//      * @brief update of the parameter Pi of the GLLiM model
//      * @details See the formula 29 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with
//      * Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
//      * @param next_theta : @see GLLiMParameters GLLiMParameters
//      * @param k : number of affine transformations
//      * @param N : the number of tuples in the data set
//      * @param r_k : the sum according to n of the posterior rnk
//      */
//     void update_Pi_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, unsigned N, double r_k);

//     /**
//      * @brief This methods verifies if the EM-algorithm has converged
//      * @details The convergence is achieved either by reaching the maximum number of iteration set in the configuration of
//      * the estimator or by reaching a likelihood enhancement less than the parameter in the configuration of the estimator.
//      * @param old_log_likelihood : The logarithm likelihood of the previous iteration
//      * @param new_log_likelihood : The logarithm likelihood of the current iteration
//      * @param current_iter : The current number of iteration passed so far
//      * @return boolean
//      */
//     bool hasConverged(double old_log_likelihood, double new_log_likelihood, unsigned current_iter);

//     /**
//      * @brief Computation stability improvement fo the covariance matrix
//      * @details This method adds a positive value to the variances in the matrix of covariance in order to improve
//      * computation stability.
//      * @tparam V : the type of covariance matrix must be a specialisation of @see Icovariance Icovariance.
//      * @param covariance : matrix covariance
//      * @param dimension : the dimension of the square matrix
//      * @param floor : the value added to the diagonal of the matrix
//      */
//     template <typename V>
//     void covStabilityImprov(V &covariance, unsigned dimension, double floor);
};


#endif // EMESTIMATOR_HPP
