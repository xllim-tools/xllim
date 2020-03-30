//
// Created by reverse-proxy on 26‏/3‏/2020.
//

#ifndef KERNELO_PREDICTORBYMEAN_H
#define KERNELO_PREDICTORBYMEAN_H

#include "../learningModel/gllim/IGLLiMLearning.h"
#include <memory>
#include "armadillo"

using namespace arma;

namespace prediction{
    class PredictorByMean {
    public:
        PredictorByMean(const std::shared_ptr<learningModel::IGLLiMLearning>& learningModel);
        void predict(const vec &y_obs, const vec& cov_obs);


    private:
        std::shared_ptr<learningModel::IGLLiMLearning> learningModel;

        vec computeMixtureMean(const vec &weights, const mat &means);
        mat computeMixtureCov(const vec &weights, const mat &means, const cube &covs);
    };
}



#endif //KERNELO_PREDICTORBYMEAN_H
