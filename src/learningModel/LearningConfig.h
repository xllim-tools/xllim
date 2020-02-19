//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_LEARNINGCONFIG_H
#define KERNELO_LEARNINGCONFIG_H

namespace learningModel{

    class LearningConfig{};

    class EMLearningConfig : public LearningConfig{
    public:
        int max_iteration;
        double ratio_ll;

        EMLearningConfig(int max_iteration, double ratio_ll){
            this->max_iteration = max_iteration;
            this->ratio_ll = ratio_ll;
        }
    };

    class GMMLearningConfig : public LearningConfig{
    public:
        int kmeans_iteration;
        int em_iteration;

        GMMLearningConfig(int kmeans_iteration, int em_iteration){
            this->kmeans_iteration = kmeans_iteration;
            this->em_iteration = em_iteration;
        }
    };

}

#endif //KERNELO_LEARNINGCONFIG_H
