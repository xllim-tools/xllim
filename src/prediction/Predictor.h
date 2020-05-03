//
// Created by reverse-proxy on 26‏/3‏/2020.
//

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
    class Predictor : public IPredictor {
    public:
        Predictor(
                const std::shared_ptr<learningModel::IGLLiMLearning>& learningModel,
                unsigned k_merged,
                unsigned k_pred_mean,
                double threshold);

        PredictionResult predict(const vec &y_obs, const vec &cov_obs) override ;
        Mat<unsigned> regularize(const cube &series) override;


    private:
        std::shared_ptr<learningModel::IGLLiMLearning> learningModel;
        unsigned k_merged = K_MERGED;
        double threshold = THRESHOLD;
        unsigned k_pred_mean = K_PRED_MEAN;

        FRIEND_TEST(PredictionByCentersTests, generatePermutations);
        FRIEND_TEST(PredictionByCentersTests, regularize);

        // compute the dissimilarity criterion of a merge and compute also that merge
        static void computeDissimilarityCriterion(
                MultivariateGaussian &g1,
                MultivariateGaussian &g2,
                MultivariateGaussian &mergedG1G2,
                double &dissimilarity
                );

        // search for best pair of gaussians to merge, merge it
        static void findPairToMerge(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians);

        // merge two gaussians
        static MultivariateGaussian mergeTwoGaussians(const MultivariateGaussian &g1, const MultivariateGaussian &g2);

        // reduce the number of gaussians before starting the merging algorithm
        void reduceGaussians(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians, unsigned &K);

        static double safeCovDet(mat &covariance);

        static Mat<unsigned> generatePermutations(unsigned N);

        vec computeMixtureMean(const vec &weights, const mat &means);

        mat computeMixtureCov(const vec &weights, const mat &means, const cube &covs);

    };
}


#endif //KERNELO_PREDICTOR_H
