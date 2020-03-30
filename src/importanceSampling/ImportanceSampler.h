//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#ifndef KERNELO_IMPORTANCESAMPLER_H
#define KERNELO_IMPORTANCESAMPLER_H

#include "IimportanceSampler.h"

namespace importanceSampling {
    class ImportanceSampler : public IimportanceSampler{
    public:
        ImportanceSampler(std::shared_ptr<DataGeneration::StatModel> statModel,
                          std::shared_ptr<ISProposition> isProposition);

        void importanceSampling() override;
        void diagnostic(mat &samples, vec &weights) override;
    };
}


#endif //KERNELO_IMPORTANCESAMPLER_H
