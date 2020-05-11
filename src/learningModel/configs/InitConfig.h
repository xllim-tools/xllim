/**
 * @file InitConfig.h
 * @brief The definition of the classes used to configure the initializer of the learning model.
 * @author Sami DJOUADI
 * @version 1.1
 * @date 03/03/2020
 */

#ifndef KERNELO_INITCONFIG_H
#define KERNELO_INITCONFIG_H

#include "LearningConfig.h"
#include "../../dataGeneration/GeneratorStrategy.h"
#include "../../dataGeneration/RandomGenerator.h"
#include <memory>
#include <utility>

namespace learningModel{

    /**
     * @class InitConfig
     *
     * @brief Common interface of the configuration classes.
     */
    class InitConfig{
    protected:
        virtual ~InitConfig() = default;
    };

    /**
     * @class FixedInitConfig
     * @brief This class wraps the parameters used to configure the @see FixedInitializer "FixedInitializer".
     */
    class FixedInitConfig: public InitConfig{
    public:
        unsigned seed; /**< Is used to generate random means to initialize the GMM of the low dimension data. */
        std::shared_ptr<GMMLearningConfig> gmmLearningConfig; /**< Configures the GMM estimator used in the intialization.
 * See the documentation of @see GMMLearningConfig "GMMLearningConfig". */
        std::shared_ptr<EMLearningConfig> emLearningConfig; /**< Configures the EM estimator used in the intialization.
 * See the documentation of @see EMLearningConfig "EMLearningConfig".*/
        std::shared_ptr<DataGeneration::GeneratorStrategy> generator;

        /**
         * Constructor
         * @param seed : unsigned
         * @param gmmLearningConfig : @see GMMLearningConfig GMMLearningConfig
         * @param emLearningConfig : @see EMLearningConfig EMLearningConfig
         */
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

    /**
     * @class MultInitConfig
     * @brief This class wraps the parameters used to configure the @see MultInitializer "MultInitializer".
     */
    class MultInitConfig: public InitConfig{
    public:
        unsigned seed; /**< Is used to generate random means to initialize the GMM of the low dimension data. */
        unsigned nb_iter_EM; /**< Is the number of iterations of the EM algorithm */
        unsigned nb_experiences;
        std::shared_ptr<GMMLearningConfig> gmmLearningConfig; /**< Configures the GMM estimator used in the intialization.
 * See the documentation of @see GMMLearningConfig "GMMLearningConfig". */
        std::shared_ptr<EMLearningConfig> emLearningConfig; /**< Configures the EM estimator used in the intialization.
 * See the documentation of @see EMLearningConfig "EMLearningConfig".*/
        std::shared_ptr<DataGeneration::GeneratorStrategy> generator;

        /**
         * Constructor
         * @param seed : unsigned
         * @param nb_iter_EM : unsigned
         * @param nb_experiences : unsigned
         * @param gmmLearningConfig : @see GMMLearningConfig GMMLearningConfig
         * @param emLearningConfig : @see EMLearningConfig EMLearningConfig
         */
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
