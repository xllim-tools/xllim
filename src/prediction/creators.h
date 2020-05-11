/**
 * @file creators.h
 * @brief Configuration structures of the prediction module
 * @author Sami DJOUADI
 * @version 1.2
 * @date 19/04/2020
 */

#ifndef KERNELO_PRED_CREATORS_H
#define KERNELO_PRED_CREATORS_H


#include "../learningModel/gllim/IGLLiMLearning.h"
#include "IPredictor.h"
#include "Predictor.h"

namespace prediction{

    /**
     * @struct PredictionConfig
     *
     * This struct wraps the parameters used to configure the prediction module. It contains the method create that
     * returns a shared pointer to a @ref IPredictor IPredictor
     */
    struct PredictionConfig{
        unsigned k_merged; /**< The number of centers to obtain while using the prediction by the centers*/
        unsigned k_pred_mean; /**< The number of gaussian distribution in the reduced GMM before returning the prediction by the mean */
        double threshold; /**< While reducing the size of the GMM during the prediction by the centers, only the components with a weight
 * superior or equal to the threshold are kept.*/
        std::shared_ptr<learningModel::IGLLiMLearning> learningModel; /**< The trained learning model used bt the prediction module*/

        /**
         * This method creates a predictor given the configuration parameters and returns a shared of it.
         * @return A shared pointer of of the created predictor..
         */
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
