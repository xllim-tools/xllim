/**
 * @file GLLiMLearning.h
 * @brief GLLiMLearning class definition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 13/02/2020
 */

#ifndef KERNELO_GLLIMLEARNING_H
#define KERNELO_GLLIMLEARNING_H

#include "IGLLiMLearning.h"
#include "../initializers/Initializers.h"
#include "../estimators/Estimators.h"
#include <memory>

namespace learningModel{

    /**
     * @class GLLiMLearning
     * @details This is the concrete class that provides various features of the GLLiM modes : Initialization , training , GMM computation
     * Conditional densities computation , GLLiM export and import... etc. The class is generic and mutates by replacing the type
     * of both the matrices of covariance in Gamma and Sigma parameters of the GLLiM model. This mutations are crucial for time
     * execution and memory storage improvement. Computations are adapted and optimized according to each type of these matrices.
     * See Icovariance.h for to understand the difference between the types of the Gamma and Sigma parameters.
     * @tparam T : the type of Gamma matrices must be a specialisation of @see Icovariance Icovariance.
     * @tparam U : the type of Sigma matrices must be a specialisation of @see Icovariance Icovariance.
     */
    template <typename T, typename U >
    class GLLiMLearning : public IGLLiMLearning {
    public:
        /**
         * Constructor
         * @param initializer : a shared pointer to the initializer of the GLLiM model @see Iinitilizer Iinitilizer.
         * @param estimator : a shared pointer to the estimator of the GLLiM model @see Iestimator Iestimator.
         * @param K : the number of affine transformation and the number of gaussian distributions in the mixture.
         */
        GLLiMLearning(std::shared_ptr<Iinitilizer<T,U>> initializer, std::shared_ptr<Iestimator<T,U>> estimator, unsigned K);

        void train(const mat &x, const mat &y) override;
        void initialize(const mat &x, const mat &y) override;
        void getModel(GLLiM &gllim) override ;
        void setModel(GLLiM &gllim) override ;
        arma::gmm_full computeGMM(const vec &y_obs, const vec &cov_obs) override ;
        void getInverse(GLLiM &gllim) override ;
        void directLogDensity(double *x, double *weights, double *means, double *covs) override ;
        void inverseLogDensity(double *y, double *weights, double *means, double *covs) override ;

        GLLiMParameters<FullCovariance, FullCovariance> inverse(GLLiMParameters<T,U> &gllim_direct);
        std::shared_ptr<GLLiMParameters<T,U>> getParameters(){
            return gllim_parameters;
        }

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, T>(), "Type U must be Icovariance specialization");

        template <typename V, typename W>
        arma::gmm_full logDensity(std::shared_ptr<GLLiMParameters<V,W>> gllim, const vec &x);

    private:
        std::shared_ptr<Iinitilizer<T,U>> initializer; /**< @see Iinitilizer Iinitilizer*/
        std::shared_ptr<Iestimator<T,U>> estimator;/**< @see Iestimator Iestimator*/
        std::shared_ptr<GLLiMParameters<T,U>> gllim_parameters; /**< The parameters of the direct GLLiM model*/
        std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> inverse_gllim_parameters; /**< The parameters of the inverse GLLiM model*/
        unsigned K; /**< the number of affine transformation and the number of gaussian distributions in the mixture */

    protected:
        /**
         * This method adjusts the Sigma parameter of the trained GLLiM model with the variance of the observation before computing the corresponding GMM
         * @param gllim : the parameters of the trained GLLiM
         * @param cov_obs : the variance or measure error of the observation
         */
        void alterCovariance(GLLiMParameters<T, U> &gllim, const vec &cov_obs);

    };

}

#include "GLLiMLearning.tpp"



#endif //KERNELO_GLLIMLEARNING_H
