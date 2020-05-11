/**
 * @file Predictor.h
 * @brief Predictor class definition
 * @author Sami DJOUADI
 * @version 1.2
 * @date 26/03/2019
 */

#ifndef KERNELO_PREDICTOR_H
#define KERNELO_PREDICTOR_H

#include "../learningModel/gllim/IGLLiMLearning.h"
#include "MultivariateGaussian.h"
#include <memory>
#include "armadillo"
#include <gtest/gtest_prod.h>
#include "PredictionResult.h"
#include "IPredictor.h"

#define K_MERGED 2
#define K_PRED_MEAN 10
#define THRESHOLD 1e-10


using namespace arma;

namespace prediction {
    /**
     * @class Predictor
     * @details This class implements the interface IPredictor. It performs the prediction by the mean and the predictin by centers.
     * It implements the regularization algorithm to get predictions more adapter to a context that requires regularity between the
     * predictions of different observations. It also implements the GMM merging algorithm using Kullback-Leibler as merging criterion.
     * Some private functions relative to merging two GMM and the computation of the mean and matrix of covariance of a GMM.
     */
    class Predictor : public IPredictor {
    public:
        /**
         * Constructor
         * @param learningModel : @see IGLLiMLearning IGLLiMLearning
         * @param k_merged : Number of gaussian distributions that are kept at the end of the GMM merging algorithm. It also defines
         * the number of centers to return after performing the prediction by the centers.
         * @param k_pred_mean : Number of gaussian distributions that are kept during th prediction by the mean.
         * @param threshold : The minimum weight of a gaussian distribution that is allowed in a GMM otherwise it is deleted.
         */
        Predictor(
                const std::shared_ptr<learningModel::IGLLiMLearning>& learningModel,
                unsigned k_merged,
                unsigned k_pred_mean,
                double threshold);

        PredictionResult predict(const vec &y_obs, const vec &cov_obs) override ;
        Mat<unsigned> regularize(const cube &series) override;


    private:
        std::shared_ptr<learningModel::IGLLiMLearning> learningModel;
        unsigned k_merged = K_MERGED; /**< Number of gaussian distributions that are kept at the end of the GMM merging algorithm. It also defines
         * the number of centers to return after performing the prediction by the centers.*/
        double threshold = THRESHOLD; /**< The minimum weight of a gaussian distribution that is allowed in a GMM otherwise it is deleted. */
        unsigned k_pred_mean = K_PRED_MEAN; /**< Number of gaussian distributions that are kept during th prediction by the mean.*/

        FRIEND_TEST(PredictionByCentersTests, generatePermutations);
        FRIEND_TEST(PredictionByCentersTests, regularize);


        /**
         * @details This method computes the Kullback-Leibler dissimilarity criterion of a merge of two gaussian distributions
         * and computes also that merge. See Kullback-Leibler Approach to Gaussian Reduction , DOI: 10.1109/TAES.2007.4383588
         * @param g1 : The first gaussian distribution of type @see MultivariateGaussian MultivariateGaussian
         * @param g2 : The second gaussian distribution of type @see MultivariateGaussian MultivariateGaussian
         * @param mergedG1G2 : The gaussian distribution that results from the merging , it is also of type @see MultivariateGaussian MultivariateGaussian
         * @param dissimilarity : The value of the dissimilarity criterion of the potential merging
         */
        static void computeDissimilarityCriterion(
                MultivariateGaussian &g1,
                MultivariateGaussian &g2,
                MultivariateGaussian &mergedG1G2,
                double &dissimilarity
                );

        /**
         * @details This methods searches for the best two gaussian to merge from a mixture of gaussian distributions. When found it calls the merging method.
         * @param gaussians : A set of gaussian distributions and a boolean which informs if the gaussian is logically deleted or not.
         */
        static void findPairToMerge(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians);

        /**
         * @details This method merges two gaussian distributions and returns the result as a gaussian distribution.
         * @param g1 : The first gaussian distribution of type @see MultivariateGaussian MultivariateGaussian
         * @param g2 : The second gaussian distribution of type @see MultivariateGaussian MultivariateGaussian
         * @return @see MultivariateGaussian MultivariateGaussian
         */
        static MultivariateGaussian mergeTwoGaussians(const MultivariateGaussian &g1, const MultivariateGaussian &g2);

        // reduce the number of gaussians before starting the merging algorithm
        /**
         * @details This method reduces the number of distributions iin the GMM before starting the merging algorithm.
         * The algorithm starts by sorting the distributions by their weights. Then it keeps the K_merged first distributions with
         * a weight higher than the threshold. The remaining distributions that fullfill the threshold conditions are merged into one
         * gaussian distribution.
         * @param gaussians : A set of gaussian distributions and a boolean which informs if the gaussian is logically deleted or not.
         * @param K : The size of the mixture after the reduction.
         */
        void reduceGaussians(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians, unsigned &K);

        static double safeCovDet(mat &covariance);

        /**
         * @details This method generate all possible permutations of a set of numbers between 0 and N-1
         * @param N
         * @return is a matrix of all the fact(N) permutations.
         */
        static Mat<unsigned> generatePermutations(unsigned N);

        /**
         * @details This method computes the mean of a GMM
         * @param weights : The weights of a GMM
         * @param means : The means of the GMM
         * @return The mean of the mixture.
         */
        vec computeMixtureMean(const vec &weights, const mat &means);

        /**
         * @details This method computes the covariance matrix of the mixture
         * @param weights : The weights of a GMM
         * @param means : The means of the GMM
         * @param covs : The covariance matrices of the GMM
         * @return The covariance matrix of the GMM
         */
        mat computeMixtureCov(const vec &weights, const mat &means, const cube &covs);

    };
}


#endif //KERNELO_PREDICTOR_H
