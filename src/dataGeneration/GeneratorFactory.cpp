//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#include "GeneratorFactory.h"
#include "SobolGenerator.h"
#include "RandomGenerator.h"
#include "LatinCubeGenerator.h"

using namespace DataGeneration;

std::shared_ptr<GeneratorStrategy> GeneratorFactory::create(std::string generatorType) {
    if(generatorType == "sobol"){
        return std::shared_ptr<GeneratorStrategy> (new SobolGenerator());
    }else if (generatorType == "latin_cube") {
        return std::shared_ptr<GeneratorStrategy> (new LatinCubeGenerator());
    }else {
        return std::shared_ptr<GeneratorStrategy> (new RandomGenerator());
    }
}

