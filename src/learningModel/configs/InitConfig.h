//
// Created by reverse-proxy on 3‏/3‏/2020.
//

#ifndef KERNELO_INITCONFIG_H
#define KERNELO_INITCONFIG_H

#include "LearningConfig.h"
#include "../../dataGeneration/GeneratorStrategy.h"
#include <memory>
#include <utility>

namespace learningModel{

    class InitConfig{};

    class FixedInitConfig: public InitConfig{
    public:
        unsigned seed;
        GMMLearningConfig gmmLearningConfig;
        EMLearningConfig emLearningConfig;
        std::shared_ptr<DataGeneration::GeneratorStrategy> generator;

        FixedInitConfig(
                unsigned seed,
                const GMMLearningConfig& gmmLearningConfig,
                const EMLearningConfig& emLearningConfig,
                const std::shared_ptr<DataGeneration::GeneratorStrategy>& generator){

            this->seed = seed;
            this->emLearningConfig = emLearningConfig;
            this->gmmLearningConfig = gmmLearningConfig;
            this->generator = generator;
        }
    };

    class MultInitConfig: public InitConfig{
    public:
        unsigned seed;
        unsigned nb_iter_EM;
        unsigned nb_experiences;
        GMMLearningConfig gmmLearningConfig;
        EMLearningConfig emLearningConfig;
        std::shared_ptr<DataGeneration::GeneratorStrategy> generator;

        MultInitConfig(
                unsigned seed,
                unsigned nb_iter_EM,
                unsigned nb_experiences,
                const GMMLearningConfig& gmmLearningConfig,
                const EMLearningConfig& emLearningConfig,
                const std::shared_ptr<DataGeneration::GeneratorStrategy>& generator){
            this->seed = seed;
            this->nb_iter_EM = nb_iter_EM;
            this->nb_experiences = nb_experiences;
            this->emLearningConfig = emLearningConfig;
            this->gmmLearningConfig = gmmLearningConfig;
            this->generator = generator;
        }
    };
}

#endif //KERNELO_INITCONFIG_H
