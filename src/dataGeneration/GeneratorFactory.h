//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_GENERATORFACTORY_H
#define KERNELO_GENERATORFACTORY_H

#include "GeneratorStrategy.h"
#include <utility>
#include <memory>

namespace DataGeneration {
    class GeneratorFactory {
    public:
        static std::shared_ptr<GeneratorStrategy> create(const std::string& generatorType);
    };
}



#endif //KERNELO_GENERATORFACTORY_H
