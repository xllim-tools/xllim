//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_GLLIMLEARNING_H
#define KERNELO_GLLIMLEARNING_H

#include "IGLLiMLearning.h"
#include "Initializers.h"
#include "Estimators.h"
#include <memory>

namespace learningModel{
    template <typename T, typename U >
    class GLLiMLearning : public IGLLiMLearning {
    public:
        GLLiMLearning(std::shared_ptr<Iinitilizer<T,U>> initializer, std::shared_ptr<Iestimator<T,U>> estimator);
        void train(mat x, mat y);
        void initialize(mat x, mat y);
        //GLLiM getModel();

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, T>(), "Type U must be Icovariance specialization");

    private:
        std::shared_ptr<Iinitilizer<T,U>> initializer;
        std::shared_ptr<Iestimator<T,U>> estimator;
        std::shared_ptr<GLLiMParameters<T,U>> gllim_parameters;
    };
}



#endif //KERNELO_GLLIMLEARNING_H
