/**
 * @file HapkeAdapter.h
 * @brief Hapke model adapter class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 14/12/2019
 */

#ifndef KERNELO_HAPKEADAPTER_H
#define KERNELO_HAPKEADAPTER_H

#include <armadillo>

using namespace arma;
namespace Functional {
/**
 * @class HapkeAdapter
 * @brief abstract class that adapt @ref HapkeModel "hapke model" to models with different
 * number of parameters for example a model with 3 parameters {omega, theta_bar, b}.
 *
 * The class contains the parameters which may be initialized like b0 and h or calculated
 * from other parameters like c, these attributes are accessible via getters.
 */
    class HapkeAdapter {
    public:

        /**
         * This method adapts the photometry to a 6 parameters model in order to have same
         * formulas in @ref HapkeModel "hapke model" for all the variants.
         *
         * Example : if using a 4 parameters model, the method sets b0 and h to default value or
         * to the arguments in the constructor of a 4 parameters adapter. And it sets C to value of C
         * in the photometry vector.
         *
         * @param photometry
         */
        virtual void adaptModel(rowvec &photometry) = 0;

        /**
         * this method returns the photometry dimension
         * @return photometry dimension
         */
        virtual int get_dimension_L() = 0;

        /**
         * b0 getter
         * @return b0
         */
        double get_b0() { return b0; };

        /**
         * h getter
         * @return h
         */
        double get_h() { return h; };

        /**
         * c getter
         * @return c
         */
        double get_c() { return c; };

    protected:
        double c; /**< fraction of the backward scattering */
        double b0; /**< amplitude of the opposition effect */
        double h; /**< angular width of the opposition effect */
    };
}

#endif //KERNELO_HAPKEADAPTER_H


