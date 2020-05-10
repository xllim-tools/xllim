/**
 * @file EstimatorFactory.h
 * @brief Factory class of the GLLiM estimator
 * @author Sami DJOUADI
 * @version 1.1
 * @date 12/02/2020
 */

#ifndef KERNELO_ESTIMATORFACTORY_H
#define KERNELO_ESTIMATORFACTORY_H

#include "estimators/Estimators.h"

namespace learningModel{
    /**
     * @class EstimatorFactory
     *
     * This class is a factory responsible of creating an estimator for the learning model. It may be GMM based estimator
     * if the config object in the parameters refers to a GMM configuration. Other wise, an estimator based on the EM
     * algorithm for the GLLim model is created.
     *
     */
    class EstimatorFactory {
    public:

        template <typename T , typename U>
        static std::shared_ptr<Iestimator<T,U>> create(const std::shared_ptr<LearningConfig>& learningConfig);
    };

}

#include "EstimatorFactory.tpp"

#endif //KERNELO_ESTIMATORFACTORY_H
