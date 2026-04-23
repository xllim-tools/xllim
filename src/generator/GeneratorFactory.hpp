// #ifndef XLLIM_GENERATORFACTORY_H
// #define XLLIM_GENERATORFACTORY_H

#include "Generator.hpp"
#include "RandomGenerator.hpp"
#include "SobolGenerator.hpp"
#include "LatinCubeGenerator.hpp"
#include <memory>
#include <stdexcept>  // for std::runtime_error

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
        throw std::runtime_error("Invalid Generator type. It must be one of the following : 'random', 'sobol, 'latin'.");  // throw by value, not pointer (S1035)
    }
}

// #endif //XLLIM_GENERATORFACTORY_H
