// #ifndef KERNELO_GENERATORFACTORY_H
// #define KERNELO_GENERATORFACTORY_H

#include "Generator.hpp"
#include "RandomGenerator.hpp"
#include "SobolGenerator.hpp"
#include "LatinCubeGenerator.hpp"
#include <memory>

using namespace DataGeneration;

inline std::shared_ptr<Generator> createGenerator(const std::string &generator_type, unsigned int seed)
{
    if (generator_type == "random")
    {
        return std::shared_ptr<Generator>(new RandomGenerator(seed));
    }
    else if (generator_type == "sobol")
    {
        return std::shared_ptr<Generator>(new SobolGenerator());
    }
    else if (generator_type == "latin_cube")
    {
        return std::shared_ptr<Generator>(new LatinCubeGenerator(seed));
    }
    else
    {
        throw "Invalid Generator type. It must be one of the following : 'random', 'sobol, 'latin'.";
    }
}

// #endif //KERNELO_GENERATORFACTORY_H
