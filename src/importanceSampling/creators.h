//
// Created by reverse-proxy on 19‏/4‏/2020.
//

#ifndef KERNELO_IS_CREATORS_H
#define KERNELO_IS_CREATORS_H

#include <memory>
#include <string>
#include <memory>
#include "../../src/dataGeneration/StatModel.h"
#include "proposition/GaussianMixtureProposition.h"
#include "proposition/GaussianRegularizedProposition.h"
#include "ImportanceSampler.h"

namespace importanceSampling{
    struct GaussianMixturePropositionConfig{
        double *weights;
        double *means;
        double *covariances;
        unsigned K;
        unsigned L;

        std::shared_ptr<ISProposition> create(){
            vec weights_arma(&weights[0], K, false, true);
            mat means_arma(&means[0],L, K, false, true);
            cube covariances_arma(&covariances[0],L,L,K, false, true);

            return std::shared_ptr<ISProposition>(
                    new GaussianMixtureProposition(
                            weights_arma,
                            means_arma,
                            covariances_arma
                            )
                    );
        }
    };

    struct GaussianRegularizedPropositionConfig{
        double *means;
        double *covariances;
        unsigned L;

        std::shared_ptr<ISProposition> create(){
            vec means_arma(&means[0],L, false, true);
            mat covs_arma(&covariances[0], L,L, false , true);
            return std::shared_ptr<ISProposition>(
                    new GaussianRegularizedProposition(
                            means_arma,
                            covs_arma
                            )
                    );
        }
    };


    struct ImportanceSamplingConfig{
        unsigned N_Samples;
        std::shared_ptr<DataGeneration::StatModel> statModel;

        std::shared_ptr<ImportanceSampler> create(){
            ISTarget target;
            target.setTarget(statModel);
            return std::make_shared<ImportanceSampler>(
                    N_Samples,
                    std::make_shared<ISTarget>(target));
        };
    };



}

#endif //KERNELO_IS_CREATORS_H
