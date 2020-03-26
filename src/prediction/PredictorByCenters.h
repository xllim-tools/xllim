//
// Created by reverse-proxy on 26‏/3‏/2020.
//

#ifndef KERNELO_PREDICTORBYCENTERS_H
#define KERNELO_PREDICTORBYCENTERS_H

#include "../learningModel/gllim/IGLLiMLearning.h"
#include "MultivariateGaussian.h"
#include <memory>
#include "armadillo"

#define K_MERGED 2
#define THRESHOLD 1e-10


using namespace arma;

namespace prediction {
    class PredictorByCenters {
    public:
        PredictorByCenters(
                const std::shared_ptr<learningModel::IGLLiMLearning>& learningModel,
                unsigned k_merged,
                double threshold);
        std::vector<vec> predict(const vec &y_obs, const vec& cov_obs);

    private:
        std::shared_ptr<learningModel::IGLLiMLearning> learningModel;
        unsigned k_merged = K_MERGED;
        double threshold = THRESHOLD;

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
        void reduceGaussians(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians);

        static double safeCovDet(mat &covariance);

    };
}


#endif //KERNELO_PREDICTORBYCENTERS_H
