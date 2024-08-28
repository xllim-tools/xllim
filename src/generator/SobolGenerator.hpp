#ifndef KERNELO_SOBOLGENERATOR_H
#define KERNELO_SOBOLGENERATOR_H

#include "Generator.hpp"

using namespace arma;

namespace DataGeneration{

    /**
     * @brief A data generator using Sobol engine
     *
     * @details this concrete strategy uses Sobol engine to generate sobol sequence while following
     * the base strategy interface.
     *
     * See Sobol engine code documentation in Boost library :
     * https://www.boost.org/doc/libs/1_72_0/boost/random/sobol.hpp
     *
     * See Sobol sequence algorithm : https://en.wikipedia.org/wiki/Sobol_sequence
     */
    class SobolGenerator : public Generator {
    public:
        void execute(mat &x) final ;
    };
}


#endif //KERNELO_SOBOLGENERATOR_H
