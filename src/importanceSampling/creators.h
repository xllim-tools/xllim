/**
 * @file creators.h
 * @brief Configuration structures of the importance sampling module
 * @author Sami DJOUADI
 * @version 1.2
 * @date 19/04/2020
 */

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

    /**
     * @struct GaussianMixturePropositionConfig
     * @details This struct wraps the parameters that configure the a proposition law of the importance sampling based on a GMM.
     * See @see GaussianMixtureProposition GaussianMixtureProposition
     */
    struct GaussianMixturePropositionConfig{
        double *weights; /**< The weights (K) of the centers*/
        double *means; /**< The centers (L,K) that stands for the predictions*/
        double *covariances; /**< The covariance matrices (L,L,K) of the centers*/
        unsigned K; /**< The number of distributions in the mixture*/
        unsigned L; /**< the number of variables*/

        /**
         * This method creates a proposition law of the importance sampling based on a GMM and returns a shared pointer of it.
         * @return std::shared_ptr<ISProposition>
         */
        std::shared_ptr<ISProposition> create(){
            vec weights_arma(&weights[0], K, false, true);
            mat means_arma(&means[0],K, L, false, true);
            cube covariances_arma(&covariances[0],L,L,K, false, true);

            means_arma = means_arma.t();

            means_arma.print("gmm_means : after import");
            covariances_arma.print("gmm_covs : after import");

            return std::shared_ptr<ISProposition>(
                    new GaussianMixtureProposition(
                            weights_arma,
                            means_arma,
                            covariances_arma
                            )
                    );
        }
    };

    /**
     * @struct GaussianRegularizedPropositionConfig
     * @details This struct wraps the parameters that configure the a proposition law of the importance sampling based on a regularized
     * gaussian distribution. See @see GaussianRegularizedProposition GaussianRegularizedProposition.
     */
    struct GaussianRegularizedPropositionConfig{
        double *means; /**< The mean of the Gaussian distribution*/
        double *covariances;/**< the covariance matrix of the gaussian distibution*/
        unsigned L; /** The number of variables of the multivariate gaussian distibution*/

        /**
         * This method creates a proposition law of the importance sampling based on a regularized gaussian distribution and returns a shared pointer of it.
         * @return std::shared_ptr<ISProposition>
         */
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

    /**
     * @struct ImportanceSamplingConfig
     * @details This struct wraps the parameters that configure the importance sampler.
     */
    struct ImportanceSamplingConfig{
        unsigned N_Samples; /**< The number of samples to generate*/
        std::shared_ptr<DataGeneration::StatModel> statModel; /**< The stat model is used to construct the target law of the importance sampler*/

        /**
         * This method creates an importance sampler and returns a pointer of it.
         * @return std::shared_ptr<ImportanceSampler>
         */
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
