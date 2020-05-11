/**
 * @file InitializerFactory.h
 * @brief Factory class of the GLLiM initializer
 * @author Sami DJOUADI
 * @version 1.1
 * @date 19/03/2020
 */

#ifndef KERNELO_INITIALIZERFACTORY_H
#define KERNELO_INITIALIZERFACTORY_H

#include "initializers/Initializers.h"
#include "configs/InitConfig.h"

namespace learningModel{

    /**
     * @class InitializerFactory
     *
     * This class is a factory responsible of creating an initializer for the learning model. The type of the created
     * initializer depends on the configuration object in the parameters of the method "create".
     * It may be @see FixedInitializer "FixedInitializer" or @see MultInitializer "MultInitializer".
     *
     */
    class InitializerFactory {
    public:

        template <typename T , typename U>
        static std::shared_ptr<Iinitilizer<T,U>> create(const std::shared_ptr<InitConfig>& initConfig);
    };

}

#include "InitializerFactory.tpp"

#endif //KERNELO_INITIALIZERFACTORY_H
