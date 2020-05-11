/**
 * @file MultInitializer.h
 * @brief MultInitializer class definition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 03/03/2020
 */

#ifndef KERNELO_MULTINITIALIZER_H
#define KERNELO_MULTINITIALIZER_H

#include "Initializers.h"
#include "../configs/InitConfig.h"
#include "../../dataGeneration/GeneratorStrategy.h"
#include <memory>

namespace learningModel{
    /**
     * @class MultInitializer
     * @details The multi experiences initialization uses a GMM then the EM algorithm to initialize theta of the GLLiM model.
     * It repeats the process nb_experiences times and in each experiences it runs the GLLiM-EM algorithm nb_iter_EM times.
     * Only the best initialization is saved based on the maximum of likelihood obtained through the experiences.
     * @tparam T : the type of Gamma matrices must be a specialisation of @see Icovariance Icovariance.
     * @tparam U : the type of Sigma matrices must be a specialisation of @see Icovariance Icovariance.
     */
    template <typename T , typename U >
    class MultInitializer : public Iinitilizer<T,U> {

    public:
        /**
         * Constructor
         * @param config : @see MultInitConfig MultInitConfig
         */
        explicit MultInitializer(const std::shared_ptr<MultInitConfig>& config);
        std::shared_ptr<GLLiMParameters <T, U>> execute(const mat &x, const mat &y, unsigned K) override;

    private:
        std::shared_ptr<MultInitConfig> config; /**< @see MultInitConfig MultInitConfig*/
    };
}

#include "MultInitializer.tpp"


#endif //KERNELO_MULTINITIALIZER_H
