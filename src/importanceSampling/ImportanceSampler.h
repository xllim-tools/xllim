//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#ifndef KERNELO_IMPORTANCESAMPLER_H
#define KERNELO_IMPORTANCESAMPLER_H

#include "proposition/ISProposition.h"
#include "ISResult.h"
#include "target/ISTarget.h"
#include "ImportanceSamplingResult.h"
#include <memory>
#include <utility>

namespace importanceSampling {

    class ImportanceSampler{
    public:

        ImportanceSampler(
                unsigned N_Samples,
                std::shared_ptr<ISTarget> isTarget);

        void execute(
                std::shared_ptr<ISProposition> isProposition,
                double *y_obs,
                double *y_cov,
                unsigned size,
                std::shared_ptr<ImportanceSamplingResult> resultExport
        );

        ISResult execute(
                std::shared_ptr<ISProposition> isProposition,
                const vec &y_obs,
                const vec &y_cov);

    private:
        unsigned N_Samples;
        std::shared_ptr<ISTarget> isTarget;

        ISDiagnostic diagnostic(
                mat &samples,
                vec &weights,
                const vec &y_obs,
                const vec &y_cov,
                std::shared_ptr<ISProposition> isProposition);
    };
}


#endif //KERNELO_IMPORTANCESAMPLER_H
