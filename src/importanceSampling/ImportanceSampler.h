//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#ifndef KERNELO_IMPORTANCESAMPLER_H
#define KERNELO_IMPORTANCESAMPLER_H

#include "ISProposition.h"
#include "ISResult.h"
#include "ISTarget.h"
#include <memory>
#include <utility>

namespace importanceSampling {
    class ImportanceSampler{
    public:
        static ISResult importanceSampling(
                const std::shared_ptr<ISTarget>& isTarget,
                const std::shared_ptr<ISProposition>& isProposition,
                const vec &y_obs,
                const vec &y_cov,
                unsigned L,
                unsigned N_samples);

    private:
        static ISDiagnostic diagnostic(
                mat &samples,
                vec &weights,
                const vec &y_obs,
                const vec &y_cov,
                const std::shared_ptr<ISProposition>& isProposition,
                const std::shared_ptr<ISTarget>& isTarget);
    };
}


#endif //KERNELO_IMPORTANCESAMPLER_H
