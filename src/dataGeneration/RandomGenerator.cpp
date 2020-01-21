//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#include <memory>
#include "RandomGenerator.h"

void DataGeneration::RandomGenerator::execute(mat &x, unsigned seed) {

    // mt19937 is a standard mersenne_twister_engine
    std::mt19937_64 engine;

    // initialize a seed using system_clock
    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //std::seed_seq ss{uint32_t(seed & 0xffffffff), uint32_t(seed>>32)};
    engine.seed(seed);

    std::uniform_real_distribution<double> unif(0, 1);
    auto ptr_to_unif = std::make_shared<std::uniform_real_distribution<double>>(unif);
    auto ptr_tp_engine = std::make_shared<std::mt19937_64>(engine);

    // generate numbers
    x.for_each([ptr_tp_engine,ptr_to_unif](mat::elem_type& val){
        val = ptr_to_unif->operator()(ptr_tp_engine.operator*());
    });
}

