/**
 * @file InitConfig.h
 * @brief The definition of the classes used to configure the initializer of the learning model.
 * @author Sami DJOUADI
 * @version 1.1
 * @date 12/02/2020
 */

#ifndef KERNELO_LEARNINGCONFIG_H
#define KERNELO_LEARNINGCONFIG_H

namespace learningModel{

    /**
     * @class InitConfig
     *
     * @brief Common interface of the configuration classes.
     */
    class LearningConfig{
    protected:
        virtual ~LearningConfig() = default;
    };

    class EMLearningConfig : public LearningConfig{
    public:
        int max_iteration;
        double ratio_ll;
        double floor;

        EMLearningConfig(int max_iteration, double ratio_ll, double floor){
            this->max_iteration = max_iteration;
            this->ratio_ll = ratio_ll;
            this->floor = floor;
        }

        EMLearningConfig(){
            this->max_iteration = 5;
            this->ratio_ll = 0.5;
            this->floor = 1e-8;
        }

        // Copy constructor
        EMLearningConfig(const EMLearningConfig &config){
            this->max_iteration = config.max_iteration;
            this->ratio_ll = config.ratio_ll;
            this->floor = config.floor;
        }

    };

    class GMMLearningConfig : public LearningConfig{
    public:
        int kmeans_iteration;
        int em_iteration;
        double floor;

        GMMLearningConfig(int kmeans_iteration, int em_iteration, double floor){
            this->kmeans_iteration = kmeans_iteration;
            this->em_iteration = em_iteration;
            this->floor = floor;
        }

        GMMLearningConfig(){
            this->kmeans_iteration = 5;
            this->em_iteration = 5;
            this->floor = 1e-8;
        }

        // Copy constructor
        GMMLearningConfig(const GMMLearningConfig &config){
            this->kmeans_iteration = config.kmeans_iteration;
            this->em_iteration = config.em_iteration;
            this->floor = config.floor;
        }
    };

}

#endif //KERNELO_LEARNINGCONFIG_H
