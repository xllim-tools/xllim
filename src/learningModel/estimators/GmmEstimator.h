/**
 * @file GmmEstimator.h
 * @brief GmmEstimator class definition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 13/02/2020
 */

#ifndef KERNELO_GMMESTIMATOR_H
#define KERNELO_GMMESTIMATOR_H

#include "Estimators.h"
#include <gtest/gtest_prod.h>

namespace learningModel{

    /**
     * @class GmmEstimator
     * @brief GMM based estimator
     * @details The estimator computes the equivalent GMM of the GLLiM model, trains the GMM and computes the GLLiM Model
     * parameters again from the trained GMM. It is used when the both the matrices of Gamma and sigma are of type @see FullCovariance FullCovariance
     */
    class GmmEstimator: public Iestimator<FullCovariance, FullCovariance>{

    public:
        /**
         * Constructor
         * @param config : @see GMMLearningConfig GMMLearningConfig
         */
        explicit GmmEstimator(const std::shared_ptr<GMMLearningConfig>& config);
        GmmEstimator();

        void execute(
                const mat& x,
                const mat& y,
                std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> initial_theta) override ;

        mat getPosterior();

        /**
         * @brief GMM training method
         * @param data : data set used to train the GMM
         * @param weights : weights of the GMM
         * @param means : means of the GMM
         * @param covariances : covariance matrices of the GMM
         */
        void train(
                const mat &data,
                const vec& weights,
                const mat &means,
                const cube &covariances);

    private:
        vec Rou; /**< The weights of the GMM equivalent to the GLLiM model.*/
        mat M; /**< The means of the GMM equivalent to the GLLiM model.*/
        cube V; /**< The covariance matrices of the GMM equivalent to the GLLiM model.*/
        mat posterior; /**< the posterior from the training of the GMM */
        std::shared_ptr<GMMLearningConfig> config; /**< The estimator configuration parameters @see GMMLearningConfig GMMLearningConfig*/

        FRIEND_TEST(GmmEstimatorTest, toGMM);
        FRIEND_TEST(GmmEstimatorTest, fromGMM);

        /**
         * @brief The method transforms a GMM to GLLiM model.
         * @details See appendix 1 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with
         * Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
         * @param K : number of components in the GMM
         * @param D : High dimension value
         * @param L : low dimension value
         * @return GLLiMParameters<FullCovariance, FullCovariance>
         */
        GLLiMParameters<FullCovariance, FullCovariance> fromGMM(int K, int D, int L);

        /**
         * @brief The method transforms a GLLiM model to an equivalent GMM.
         * @details See appendix 1 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression with
         * Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
         * @param theta
         */
        void toGMM(const std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>>& theta);
    };
}

#include "GmmEstimator.tpp"

#endif //KERNELO_GMMESTIMATOR_H
