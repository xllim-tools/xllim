// #include <memory>
#include "RandomGenerator.hpp"

DataGeneration::RandomGenerator::RandomGenerator(unsigned int seed) {
    this->seed = seed;
}

void DataGeneration::RandomGenerator::execute(mat &x) {

    // mt19937 is a standard mersenne_twister_engine
    std::mt19937_64 engine;

    // initialize a seed using system_clock
    engine.seed(seed);

    std::uniform_real_distribution<double> unif(0, 1);
    auto ptr_to_unif = std::make_shared<std::uniform_real_distribution<double>>(unif);
    auto ptr_tp_engine = std::make_shared<std::mt19937_64>(engine);

    // generate numbers
    x.for_each([ptr_tp_engine,ptr_to_unif](mat::elem_type& val){
        val = ptr_to_unif->operator()(ptr_tp_engine.operator*());
    });
    seed = engine();
}

