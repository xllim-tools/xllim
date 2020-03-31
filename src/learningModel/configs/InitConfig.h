//
// Created by reverse-proxy on 3‏/3‏/2020.
//

#ifndef KERNELO_INITCONFIG_H
#define KERNELO_INITCONFIG_H

#include "LearningConfig.h"
#include "../../dataGeneration/GeneratorStrategy.h"
#include "../../dataGeneration/RandomGenerator.h"
#include <memory>
#include <utility>

namespace learningModel{

    class InitConfig{
    protected:
        virtual ~InitConfig() = default;
    };

    class FixedInitConfig: public InitConfig{
    public:
        unsigned seed;
        std::shared_ptr<GMMLearningConfig> gmmLearningConfig;
        std::shared_ptr<EMLearningConfig> emLearningConfig;
        std::shared_ptr<DataGeneration::GeneratorStrategy> generator;

        FixedInitConfig(
                unsigned seed,
                const std::shared_ptr<LearningConfig>& gmmLearningConfig,
                const std::shared_ptr<LearningConfig>& emLearningConfig){

            this->seed = seed;
            this->emLearningConfig = std::dynamic_pointer_cast<EMLearningConfig>(emLearningConfig);
            this->gmmLearningConfig = std::dynamic_pointer_cast<GMMLearningConfig>(gmmLearningConfig);
            this->generator = std::shared_ptr<DataGeneration::GeneratorStrategy> (new DataGeneration::RandomGenerator(seed));
        }
    };

    class MultInitConfig: public InitConfig{
    public:
        unsigned seed;
        unsigned nb_iter_EM;
        unsigned nb_experiences;
        std::shared_ptr<GMMLearningConfig> gmmLearningConfig;
        std::shared_ptr<EMLearningConfig> emLearningConfig;
        std::shared_ptr<DataGeneration::GeneratorStrategy> generator;

        MultInitConfig(
                unsigned seed,
                unsigned nb_iter_EM,
                unsigned nb_experiences,
                const std::shared_ptr<LearningConfig>& gmmLearningConfig,
                const std::shared_ptr<LearningConfig>& emLearningConfig){
            this->seed = seed;
            this->nb_iter_EM = nb_iter_EM;
            this->nb_experiences = nb_experiences;
            this->emLearningConfig = std::dynamic_pointer_cast<EMLearningConfig>(emLearningConfig);
            this->gmmLearningConfig = std::dynamic_pointer_cast<GMMLearningConfig>(gmmLearningConfig);
            this->generator = std::shared_ptr<DataGeneration::GeneratorStrategy> (new DataGeneration::RandomGenerator(seed));
        }
    };
}

#endif //KERNELO_INITCONFIG_H
