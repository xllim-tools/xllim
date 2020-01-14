//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#include "SobolGenerator.h"
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

using namespace DataGeneration;
using namespace arma;
using namespace boost::random;

typedef sobol_engine< boost::uint_least64_t, 64u, default_sobol_table > Sobol;

void SobolGenerator::execute(int n, int dimension, mat &x) {

    // Initialize the engine to draw randomness out of thin air
    Sobol engine(dimension);

    std::uniform_real_distribution<double> unif(0, 1);

    // generate numbers
    for (unsigned i=0; i<n; i++)
    {
        for(unsigned j=0; j<dimension; j++){
            x(i,j) = unif(engine);
        }
    }

}
