//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_ESTIMATORFACTORY_H
#define KERNELO_ESTIMATORFACTORY_H

#include "Estimators.h"

namespace learningModel{

    class EstimatorFactory {
    public:

        template <typename T = Icovariance, typename U = Icovariance>
        static Iestimator<T,U> create(LearningConfig config);
    };

}



#endif //KERNELO_ESTIMATORFACTORY_H
