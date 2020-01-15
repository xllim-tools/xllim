//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#include "RandomGenerator.h"

void DataGeneration::RandomGenerator::execute(int n, int dimension, double *x) {

    std::uniform_real_distribution<double> unif(0, 1);

    // mt19937 is a standard mersenne_twister_engine
    std::mt19937_64 engine;

    // initialize a seed using system_clock
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(seed & 0xffffffff), uint32_t(seed>>32)};
    engine.seed(ss);


    // generate random numbers
    for (unsigned i=0; i<n; i++)
    {
        for(unsigned j=0; j<dimension; j++){
            x[i*dimension+j] = unif(engine);
        }
    }
}
