//
// Created by reverse-proxy on 6‏/1‏/2020.
//

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <iostream>
#include <chrono>

#include "src/dataGeneration/LatinCubeGenerator.h"
#include "src/dataGeneration/LatinCubeGenerator.cpp"

using namespace boost::random;
typedef sobol_engine< boost::uint_least64_t, 64u, default_sobol_table > Sobol;

int main(){
    static const std::size_t dimension = 6;

    // Create a generator
    typedef boost::variate_generator<Sobol, boost::uniform_01<double>> quasi_random_gen_t;

    // Initialize the engine to draw randomness out of thin air
    Sobol engine(dimension);

    std::vector<double> sample(dimension);

    // At this point you can use std::generate, generate member f-n, etc.
    //engine.generate(sample.begin(), sample.end());


    //-------------------------------------------------------
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(seed & 0xffffffff), uint32_t(seed>>32)};

    // mt19937 is a standard mersenne_twister_engine
    std::mt19937_64 generator;
    generator.seed(ss);

    std::uniform_real_distribution<double> unif(0, 1);
    // ready to generate random numbers
    for (int i = 0; i < 20; i++)
    {
        for(int j=0; j<6; j++){
            //double currentRandomNumber = unif(generator);
            //std::cout << currentRandomNumber << " ";
        }
        //std::cout << std::endl;
    }

    // LATIN HYPERCUBE
    //DataGeneration::LatinCubeGenerator::

    int seed_latin = 123456789;
    int m = 6;
    int i;
    int j;
    int k;
    int kk;
    int n = 20;
    double *x;

    std::cout << "\n";
    std::cout << "TEST01\n";
    std::cout << "  LATIN_RANDOM chooses a Latin Square cell arrangement,\n";
    std::cout << "  and then chooses a random point from each cell.\n";
    std::cout << "\n";
    std::cout << "  Spatial dimension = " << m << "\n";
    std::cout << "  Number of points =  " << n << "\n";
    std::cout << "  Initial seed for UNIFORM = " << seed << "\n";

    x = DataGeneration::LatinCubeGenerator::latin_random_new ( m, n, seed_latin );

    DataGeneration::LatinCubeGenerator::r8mat_transpose_print ( m, n, x, "  Latin Random Square:" );

    delete [] x;



}

