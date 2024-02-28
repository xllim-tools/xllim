#include "GeneratorFactory.hpp"
#include "RandomGenerator.hpp"
// #include "SobolGenerator.hpp"
// #include "LatinCubeGenerator.hpp"

using namespace DataGeneration;

std::shared_ptr<Generator> GeneratorFactory::create(const std::string &generator_type, unsigned int seed)
{
    if (generator_type == "random")
    {
        return std::shared_ptr<Generator>(new RandomGenerator(seed));
    }
    // else if (generator_type == "sobol")
    // {
    //     return std::shared_ptr<GeneratorStrategy>(new SobolGenerator());
    // }
    // else if (generator_type == "latin_cube")
    // {
    //     return std::shared_ptr<GeneratorStrategy>(new LatinCubeGenerator(seed));
    // }
    else
    {
        throw "Invalid Generator type. It must be one of the following : 'random', 'sobol, 'latin'.";
    }
}
