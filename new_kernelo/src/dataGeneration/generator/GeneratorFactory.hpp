#ifndef KERNELO_GENERATORFACTORY_H
#define KERNELO_GENERATORFACTORY_H

#include "Generator.hpp"
// #include <utility>
// #include <memory>

namespace DataGeneration {
    /**
     * @brief This class is a simple factory responsible for creating a concrete @ref Generator
     * "data generator strategy" based on the type requested.
     */
    class GeneratorFactory {
    public:
        static std::shared_ptr<Generator> create(const std::string& generator_type, unsigned int seed);
    };
}



#endif //KERNELO_GENERATORFACTORY_H
