/**
 * @file ImportanceSampler.h
 * @brief ImportanceSampler calss definition
 * @author Sami DJOUADI
 * @version 1.2
 * @date 29/03/2020
 */

#ifndef KERNELO_IMPORTANCESAMPLER_H
#define KERNELO_IMPORTANCESAMPLER_H

#include "proposition/ISProposition.h"
#include "ISResult.h"
#include "target/ISTarget.h"
#include "ImportanceSamplingResult.h"
#include <memory>
#include <utility>

namespace importanceSampling {
    /**
     * @class ImportanceSampler
     * @details This class performs the importance sampling algorithm of a prediction given a target and proposition density.
     * The class offers two format of the this method, the first one for external calls from third party language API and the
     * second one for internal calls because it uses the data structures of the library armadillo.
     */
    class ImportanceSampler{
    public:

        /**
         * Constructor
         * @param N_Samples : Number of samples of the proposition law that will be generated
         * @param isTarget : A shared pointer to the target low , see @see ISTarget ISTarget
         */
        ImportanceSampler(
                unsigned N_Samples,
                std::shared_ptr<ISTarget> isTarget);

        /**
         * This methods performs the importance sampling algorithm and returns the result in fifth parameter.
         * @param isProposition : @see ISProposition ISProposition
         * @param y_obs : a pointer to the vector of variables describing the observation
         * @param y_cov : a pointer to the vector of measure errors of the observation
         * @param size : the number of variables of the observation
         * @param resultExport : @see ImportanceSamplingResult
         */
        void execute(
                std::shared_ptr<ISProposition> isProposition,
                double *y_obs,
                double *y_cov,
                unsigned size,
                std::shared_ptr<ImportanceSamplingResult> resultExport
        );

        /**
         * This methods performs the importance sampling algorithm and returns the result as @see ISResult ISResult.
         * @param isProposition : @see ISProposition ISProposition
         * @param y_obs : A vector of variables describing the observation
         * @param y_cov : A vector of measure errors of the observation
         * @return @see ISResult ISResult
         */
        ISResult execute(
                std::shared_ptr<ISProposition> isProposition,
                const vec &y_obs,
                const vec &y_cov);

    private:
        unsigned N_Samples;/**< Number of samples of the proposition law that will be generated*/
        std::shared_ptr<ISTarget> isTarget;/**< A shared pointer to the target low , see @see ISTarget ISTarget*/

        ISDiagnostic diagnostic(
                mat &samples,
                vec &weights,
                const vec &y_obs,
                const vec &y_cov,
                std::shared_ptr<ISProposition> isProposition);
    };
}


#endif //KERNELO_IMPORTANCESAMPLER_H
