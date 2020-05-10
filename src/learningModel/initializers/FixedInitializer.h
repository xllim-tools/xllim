/**
 * @file FixedInitializer.h
 * @brief FixedInitializer class definition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 23/03/2020
 */

#ifndef KERNELO_FIXEDINITIALIZER_H
#define KERNELO_FIXEDINITIALIZER_H

#include "Initializers.h"
#include "../configs/InitConfig.h"
#include "../../dataGeneration/GeneratorStrategy.h"
#include <memory>

namespace learningModel{

    /**
     * @class FixedInitializer
     * @details The fixed initialization uses a GMM which is initialized with random and fixed values to compute the
     * initial theta of the GLLiM model.
     * @tparam T : the type of Gamma matrices must be a specialisation of @see Icovariance Icovariance.
     * @tparam U : the type of Sigma matrices must be a specialisation of @see Icovariance Icovariance.
     */
    template <typename T , typename U >
    class FixedInitializer : public Iinitilizer<T,U> {

    public:
        /**
         * Constructor
         * @param config : @see FixedInitConfig FixedInitConfig
         */
        explicit FixedInitializer(const std::shared_ptr<FixedInitConfig>& config);
        std::shared_ptr<GLLiMParameters<T, U>> execute(const mat &x, const mat &y, unsigned K) override ;

    private:
        std::shared_ptr<FixedInitConfig> config; /**< @see FixedInitConfig FixedInitConfig*/
    };
}

#include "FixedInitializer.tpp"


#endif //KERNELO_FIXEDINITIALIZER_H
