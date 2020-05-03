//
// Created by reverse-proxy on 19‏/4‏/2020.
//

#ifndef KERNELO_PRED_CREATORS_H
#define KERNELO_PRED_CREATORS_H


#include "../learningModel/gllim/IGLLiMLearning.h"
#include "IPredictor.h"
#include "Predictor.h"

namespace prediction{
    struct PredictionConfig{
        unsigned k_merged;
        unsigned k_pred_mean;
        double threshold;
        std::shared_ptr<learningModel::IGLLiMLearning> learningModel;

        std::shared_ptr<IPredictor> create(){

            return std::shared_ptr<IPredictor>( new Predictor(
                    learningModel,
                    k_merged,
                    k_pred_mean,
                    threshold));
        }
    };
}

#endif //KERNELO_PRED_CREATORS_H
