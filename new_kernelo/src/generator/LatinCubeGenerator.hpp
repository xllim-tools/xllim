#ifndef KERNELO_LATINCUBEGENERATOR_H
#define KERNELO_LATINCUBEGENERATOR_H

#include "Generator.hpp"

using namespace arma;

namespace DataGeneration{

    /**
     * @brief A Latin Hyper cube algorithm for data generation
     *
     * @details this concrete strategy implements the latin hyper cube algorithm while following
     * the base strategy interface.
     *
     * See http://people.math.sc.edu/Burkardt/cpp_src/latin_random/latin_random.html
     */
    class LatinCubeGenerator : public Generator{
    public:
        void execute(mat &x) final;
        explicit LatinCubeGenerator(unsigned seed);
    private:
        unsigned seed;
        static int i4_uniform_ab ( int ilo, int ihi, int &seed );
        static double *latin_random_new ( int dim_num, int point_num, int &seed );
        static int *perm_uniform_new ( int n, int &seed );
        static double *r8mat_uniform_01_new ( int m, int n, int &seed );
    };
}



#endif //KERNELO_LATINCUBEGENERATOR_H
