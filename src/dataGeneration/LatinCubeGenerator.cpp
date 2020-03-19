/**
 * @file LatinCubeGenerator.cpp
 * @brief LatinCubeGenerator class implementation
 * @authors Sami DJOUADI & John Burkardt
 * @version 1.0
 * @date 13/01/2020
 */

#include "LatinCubeGenerator.h"

# include <cstdlib>
# include <iostream>
# include <cmath>
# include <ctime>

using namespace std;
using namespace DataGeneration;

LatinCubeGenerator::LatinCubeGenerator(unsigned seed) {
    this->seed = seed;
}


void LatinCubeGenerator::execute(mat &x) {

    double *generated_x;

    generated_x = DataGeneration::LatinCubeGenerator::latin_random_new (x.n_cols, x.n_rows,
            reinterpret_cast<int &>(seed));

    for(unsigned i=0; i<x.n_rows ; i++){
        for(unsigned j=0; j<x.n_cols; j++){
            x(i,j) = generated_x[i*x.n_cols+j];
        }
    }
}

//****************************************************************************80

int LatinCubeGenerator::i4_uniform_ab ( int a, int b, int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    I4_UNIFORM_AB returns a scaled pseudorandom I4 between A and B.
//
//  Discussion:
//
//    The pseudorandom number should be uniformly distributed
//    between A and B.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    02 October 2012
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Paul Bratley, Bennett Fox, Linus Schrage,
//    A Guide to Simulation,
//    Second Edition,
//    Springer, 1987,
//    ISBN: 0387964673,
//    LC: QA76.9.C65.B73.
//
//    Bennett Fox,
//    Algorithm 647:
//    Implementation and Relative Efficiency of Quasirandom
//    Sequence Generators,
//    ACM Transactions on Mathematical Software,
//    Volume 12, Number 4, December 1986, pages 362-376.
//
//    Pierre L'Ecuyer,
//    Random Number Generation,
//    in Handbook of Simulation,
//    edited by Jerry Banks,
//    Wiley, 1998,
//    ISBN: 0471134031,
//    LC: T57.62.H37.
//
//    Peter Lewis, Allen Goodman, James Miller,
//    A Pseudo-Random Number Generator for the System/360,
//    IBM Systems Journal,
//    Volume 8, Number 2, 1969, pages 136-143.
//
//  Parameters:
//
//    Input, int A, B, the limits of the interval.
//
//    Input/output, int &SEED, the "seed" value, which should NOT be 0.
//    On output, SEED has been updated.
//
//    Output, int I4_UNIFORM, a number between A and B.
//
{
    int c;
    const int i4_huge = 2147483647;
    int k;
    float r;
    int value;

    if ( seed == 0 )
    {
        cerr << "\n";
        cerr << "I4_UNIFORM_AB - Fatal error!\n";
        cerr << "  Input value of SEED = 0.\n";
        exit ( 1 );
    }
//
//  Guarantee A <= B.
//
    if ( b < a )
    {
        c = a;
        a = b;
        b = c;
    }

    k = seed / 127773;

    seed = 16807 * ( seed - k * 127773 ) - k * 2836;

    if ( seed < 0 )
    {
        seed = seed + i4_huge;
    }

    r = ( float ) ( seed ) * 4.656612875E-10;
//
//  Scale R to lie between A-0.5 and B+0.5.
//
    r = ( 1.0 - r ) * ( ( float ) a - 0.5 )
        +         r   * ( ( float ) b + 0.5 );
//
//  Use rounding to convert R to an integer between A and B.
//
    value = round ( r );
//
//  Guarantee A <= VALUE <= B.
//
    if ( value < a )
    {
        value = a;
    }
    if ( b < value )
    {
        value = b;
    }

    return value;
}
//****************************************************************************80

double *LatinCubeGenerator::latin_random_new ( int dim_num, int point_num, int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    LATIN_RANDOM_NEW returns points in a Latin Random square.
//
//  Discussion:
//
//    In each spatial dimension, there will be exactly one
//    point whose coordinate value lies between consecutive
//    values in the list:
//
//      ( 0, 1, 2, ..., point_num ) / point_num
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 April 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int DIM_NUM, the spatial dimension.
//
//    Input, int POINT_NUM, the number of points.
//
//    Input/output, int &SEED, a seed for UNIFORM.
//
//    Output, double LATIN_RANDOM_NEW[DIM_NUM,POINT_NUM], the points.
//
{
    int i;
    int j;
    int *perm;
    double r;
    double *x;

    x = LatinCubeGenerator::r8mat_uniform_01_new ( dim_num, point_num, seed );
//
//  For spatial dimension I,
//    pick a random permutation of 1 to POINT_NUM,
//    force the corresponding I-th components of X to lie in the
//    interval ( PERM[J]-1, PERM[J] ) / POINT_NUM.
//
    for ( i = 0; i < dim_num; i++ )
    {
        perm = perm_uniform_new ( point_num, seed );

        for ( j = 0; j < point_num; j++ )
        {
            x[i+j*dim_num] = ( ( ( double ) perm[j] ) + x[i+j*dim_num] )
                             / ( ( double ) point_num );
        }
        delete [] perm;
    }
    return x;
}
//****************************************************************************80

int * LatinCubeGenerator::perm_uniform_new ( int n, int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    PERM_UNIFORM_NEW selects a random permutation of N objects.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 February 2014
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Albert Nijenhuis, Herbert Wilf,
//    Combinatorial Algorithms,
//    Academic Press, 1978, second edition,
//    ISBN 0-12-519260-6.
//
//  Parameters:
//
//    Input, int N, the number of objects to be permuted.
//
//    Input/output, int &SEED, a seed for the random number generator.
//
//    Output, int PERM_UNIFORM_NEW[N], a permutation of
//    (0, 1, ..., N-1).
//
{
    int i;
    int j;
    int k;
    int *p;

    p = new int[n];

    for ( i = 0; i < n; i++ )
    {
        p[i] = i;
    }

    for ( i = 0; i < n - 1; i++ )
    {
        j = i4_uniform_ab ( i, n - 1, seed );
        k    = p[i];
        p[i] = p[j];
        p[j] = k;
    }

    return p;
}
//****************************************************************************80

double * LatinCubeGenerator::r8mat_uniform_01_new ( int m, int n, int &seed )

//****************************************************************************80
//
//  Purpose:
//
//    R8MAT_UNIFORM_01_NEW returns a unit pseudorandom R8MAT.
//
//  Discussion:
//
//    An R8MAT is a doubly dimensioned array of R8's,  stored as a vector
//    in column-major order.
//
//    This routine implements the recursion
//
//      seed = 16807 * seed mod ( 2^31 - 1 )
//      unif = seed / ( 2^31 - 1 )
//
//    The integer arithmetic never requires more than 32 bits,
//    including a sign bit.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 October 2005
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Paul Bratley, Bennett Fox, Linus Schrage,
//    A Guide to Simulation,
//    Springer Verlag, pages 201-202, 1983.
//
//    Bennett Fox,
//    Algorithm 647:
//    Implementation and Relative Efficiency of Quasirandom
//    Sequence Generators,
//    ACM Transactions on Mathematical Software,
//    Volume 12, Number 4, pages 362-376, 1986.
//
//    Philip Lewis, Allen Goodman, James Miller,
//    A Pseudo-Random Number Generator for the System/360,
//    IBM Systems Journal,
//    Volume 8, pages 136-143, 1969.
//
//  Parameters:
//
//    Input, int M, N, the number of rows and columns.
//
//    Input/output, int &SEED, the "seed" value.  Normally, this
//    value should not be 0, otherwise the output value of SEED
//    will still be 0, and R8_UNIFORM will be 0.  On output, SEED has
//    been updated.
//
//    Output, double R8MAT_UNIFORM_01_NEW[M*N], a matrix of pseudorandom values.
//
{
    int i;
    const int i4_huge = 2147483647;
    int j;
    int k;
    double *r;

    r = new double[m*n];

    for ( j = 0; j < n; j++ )
    {
        for ( i = 0; i < m; i++ )
        {
            k = seed / 127773;

            seed = 16807 * ( seed - k * 127773 ) - k * 2836;

            if ( seed < 0 )
            {
                seed = seed + i4_huge;
            }
            r[i+j*m] = ( double ) ( seed ) * 4.656612875E-10;
        }
    }

    return r;
}


//****************************************************************************80