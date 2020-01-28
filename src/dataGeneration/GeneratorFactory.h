/**
 * @file GeneratorFactory.h
 * @brief GeneratorFactory class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 13/01/2020
 */

#ifndef KERNELO_GENERATORFACTORY_H
#define KERNELO_GENERATORFACTORY_H

#include "GeneratorStrategy.h"
#include <utility>
#include <memory>

namespace DataGeneration {
    /**
     * @brief This class is a simple factory responsible for creating a concrete @ref GeneratorStrategy
     * "data generator strategy" based on the type requested.
     */
    class GeneratorFactory {
    public:
        static std::shared_ptr<GeneratorStrategy> create(const std::string& generatorType);
    };
}



#endif //KERNELO_GENERATORFACTORY_H
