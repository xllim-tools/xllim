/**
 * @file SobolGenerator.cpp
 * @brief SobolGenerator class implementation
 * @author Sami DJOUADI
 * @version 1.0
 * @date 13/01/2020
 */

#include "SobolGenerator.h"
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>

using namespace DataGeneration;
using namespace arma;
using namespace boost::random;

typedef sobol_engine< boost::uint_least64_t, 64u, default_sobol_table > Sobol;

void SobolGenerator::execute(mat &x) {

    // Initialize the engine to draw randomness out of thin air
    Sobol engine(x.n_cols);

    std::uniform_real_distribution<double> unif(0, 1);

    // generate numbers
    for(unsigned i=0; i<x.n_rows ; i++){
        for(unsigned j=0; j<x.n_cols; j++){
            x(i,j) = unif(engine);
        }
    }

}
