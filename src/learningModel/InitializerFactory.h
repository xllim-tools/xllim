//
// Created by reverse-proxy on 19‏/3‏/2020.
//

#ifndef KERNELO_INITIALIZERFACTORY_H
#define KERNELO_INITIALIZERFACTORY_H

#include "initializers/Initializers.h"
#include "configs/InitConfig.h"

namespace learningModel{

    class InitializerFactory {
    public:

        template <typename T , typename U>
        static std::shared_ptr<Iinitilizer<T,U>> create(const std::shared_ptr<InitConfig>& initConfig);
    };

}

#include "InitializerFactory.tpp"

#endif //KERNELO_INITIALIZERFACTORY_H
