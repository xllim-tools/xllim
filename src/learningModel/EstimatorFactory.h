//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_ESTIMATORFACTORY_H
#define KERNELO_ESTIMATORFACTORY_H

#include "estimators/Estimators.h"

namespace learningModel{

    class EstimatorFactory {
    public:

        template <typename T , typename U>
        static std::shared_ptr<Iestimator<T,U>> create(const std::shared_ptr<LearningConfig>& learningConfig);
    };

}

#include "EstimatorFactory.tpp"



#endif //KERNELO_ESTIMATORFACTORY_H
