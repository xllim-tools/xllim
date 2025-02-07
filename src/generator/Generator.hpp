#ifndef XLLIM_GENERATOR_H
#define XLLIM_GENERATOR_H

#include <armadillo>

using namespace arma;

namespace DataGeneration
{
    /**
     * @brief DataGeneration interface
     *
     * @details the strategy interface declares operations common to all supported versions of
     * data generators. The client class uses this interface to call the algorithm defined by
     * the concrete strategies. the interface makes the concrete data generators interchangeable
     * in the client class.
     */
    class Generator
    {
    public:
        virtual void execute(mat &x) = 0;
    };
}

#endif // XLLIM_GENERATOR_H