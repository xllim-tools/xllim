//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_LATINCUBEGENERATOR_H
#define KERNELO_LATINCUBEGENERATOR_H

#include <string>


namespace DataGeneration{
    class LatinCubeGenerator{
    public:
        static int get_seed ( );
        static int i4_uniform_ab ( int ilo, int ihi, int &seed );
        static void i4vec_print ( int n, int a[], std::string title );
        static double *latin_random_new ( int dim_num, int point_num, int &seed );
        static int *perm_uniform_new ( int n, int &seed );
        static void r8mat_transpose_print ( int m, int n, double a[], std::string title );
        static void r8mat_transpose_print_some ( int m, int n, double a[], int ilo, int jlo,
                                          int ihi, int jhi, std::string title );
        static double *r8mat_uniform_01_new ( int m, int n, int &seed );
        static void r8mat_write ( std::string output_filename, int m, int n, double table[] );
        static void timestamp ( );
    };
}



#endif //KERNELO_LATINCUBEGENERATOR_H
