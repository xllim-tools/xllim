//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_LATINCUBEGENERATOR_H
#define KERNELO_LATINCUBEGENERATOR_H

#include <string>
#include "GeneratorStrategy.h"


namespace DataGeneration{
    class LatinCubeGenerator : public GeneratorStrategy{
    public:
        void execute(int n, int dimension, double *x) final;
    private:
        static int get_seed ();
        static int i4_uniform_ab ( int ilo, int ihi, int &seed );
        static double *latin_random_new ( int dim_num, int point_num, int &seed );
        static int *perm_uniform_new ( int n, int &seed );
        static double *r8mat_uniform_01_new ( int m, int n, int &seed );
    };
}



#endif //KERNELO_LATINCUBEGENERATOR_H
