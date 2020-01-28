/**
 * @file GeneratorFactory.cpp
 * @brief GeneratorFactory class implementation
 * @author Sami DJOUADI
 * @version 1.0
 * @date 13/01/2020
 */

#include "GeneratorFactory.h"
#include "SobolGenerator.h"
#include "RandomGenerator.h"
#include "LatinCubeGenerator.h"

using namespace DataGeneration;

std::shared_ptr<GeneratorStrategy> GeneratorFactory::create(const std::string& generatorType) {
    if(generatorType == "sobol"){
        return std::shared_ptr<GeneratorStrategy> (new SobolGenerator());
    }else if (generatorType == "latin_cube") {
        return std::shared_ptr<GeneratorStrategy> (new LatinCubeGenerator());
    }else {
        return std::shared_ptr<GeneratorStrategy> (new RandomGenerator());
    }
}

