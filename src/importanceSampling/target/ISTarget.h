//
// Created by reverse-proxy on 2‏/4‏/2020.
//

#ifndef KERNELO_ISTARGET_H
#define KERNELO_ISTARGET_H

#include <memory>
#include <utility>
#include "../../dataGeneration/StatModel.h"

namespace importanceSampling{
    struct ISTarget{
    private:
        std::shared_ptr<DataGeneration::StatModel> target;
    public:
        virtual double target_log_density(const vec &x, const vec &y, const vec &y_cov){
            return target->density_X_Y(x, y, y_cov);
        }

        void setTarget(std::shared_ptr<DataGeneration::StatModel> targetDistribution){
            target = std::move(targetDistribution);
        };
    };
}

#endif //KERNELO_ISTARGET_H
