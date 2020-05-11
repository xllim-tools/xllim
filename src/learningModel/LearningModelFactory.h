/**
 * @file LearningModelFactory.h
 * @brief Factory class of the learning model
 * @author Sami DJOUADI
 * @version 1.1
 * @date 19/03/2020
 */

#ifndef KERNELO_LEARNINGMODELFACTORY_H
#define KERNELO_LEARNINGMODELFACTORY_H

#include "gllim/IGLLiMLearning.h"
#include "configs/InitConfig.h"
#include "configs/LearningConfig.h"
#include <memory>

namespace learningModel{

    /**
     * @class LearningModelFactory
     *
     * This class is a factory responsible of creating an instance of the GLLiM model class. It is configured with the type
     * of the matrices of covariance Gamma and Sigma, and with the configuration objects of the initializer and the estimator
     * of the model.
     *
     */
    class LearningModelFactory {
    public:
        static std::shared_ptr<IGLLiMLearning> create(
                unsigned k,
                const std::string &GammaType,
                const std::string &SigmaType,
                const std::shared_ptr<InitConfig>& initConfig,
                const std::shared_ptr<LearningConfig>& learningConfig);
    };
}

#include "LearningModelFactory.tpp"



#endif //KERNELO_LEARNINGMODELFACTORY_H
