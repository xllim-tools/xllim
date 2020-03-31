//
// Created by reverse-proxy on 19‏/3‏/2020.
//

#ifndef KERNELO_LEARNINGMODELFACTORY_H
#define KERNELO_LEARNINGMODELFACTORY_H

#include "gllim/IGLLiMLearning.h"
#include "configs/InitConfig.h"
#include "configs/LearningConfig.h"
#include <memory>



namespace learningModel{
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
