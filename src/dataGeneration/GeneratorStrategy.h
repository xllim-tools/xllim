/**
 * @file GeneratorStrategy.h
 * @brief An interface of data generators
 * @author Sami DJOUADI
 * @version 1.0
 * @date 13/01/2020
 */

#ifndef KERNELO_GENERATORSTRATEGY_H
#define KERNELO_GENERATORSTRATEGY_H

#include <armadillo>

using namespace arma;

namespace DataGeneration{
    /**
     * @brief DataGeneration interface
     *
     * @details the strategy interface declares operations common to all supported versions of
     * data generators. The client class uses this interface to call the algorithm defined by
     * the concrete strategies. the interface makes the concrete data generators interchangeable
     * in the client class.
     */
    class GeneratorStrategy{
    public:
        virtual void execute(mat &x, unsigned seed) = 0;
    };
}

#endif //KERNELO_GENERATORSTRATEGY_H