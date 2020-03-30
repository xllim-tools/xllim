//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#ifndef KERNELO_IIMPORTANCESAMPLER_H
#define KERNELO_IIMPORTANCESAMPLER_H

#include <memory>
#include "../dataGeneration/StatModel.h"
#include "ISProposition.h"

namespace importanceSampling{
    class IimportanceSampler{
    public:
        IimportanceSampler(
                std::shared_ptr<DataGeneration::StatModel> statModel,
                std::shared_ptr<ISProposition> isProposition){
            this->statModel = statModel;
            this->isProposition = isProposition;
        }

        virtual void importanceSampling() = 0;
        virtual void diagnostic(mat &samples, vec &weights) = 0;

    protected:
        std::shared_ptr<DataGeneration::StatModel> statModel;
        std::shared_ptr<ISProposition> isProposition;
    };
};

#endif //KERNELO_IIMPORTANCESAMPLER_H
