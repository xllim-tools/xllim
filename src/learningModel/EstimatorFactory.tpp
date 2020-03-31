//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#include "configs/LearningConfig.h"
#include "estimators/GmmEstimator.h"
#include "estimators/EmEstimator.h"

using namespace learningModel;

template<typename T, typename U>
std::shared_ptr<Iestimator<T,U>> EstimatorFactory::create(const std::shared_ptr<LearningConfig>& learningConfig) {

    static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
    static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");


    return std::make_shared<EmEstimator<T,U>>(
            EmEstimator<T,U>(std::dynamic_pointer_cast<EMLearningConfig>(learningConfig))
                    );

}
