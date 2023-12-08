#ifndef KERNELO_FUNCTIONNALMODEL_H
#define KERNELO_FUNCTIONNALMODEL_H

#include <armadillo>
#include <utility>
#include <memory>

using namespace arma;

namespace Functional
{
    /**
     * @class FunctionalModel
     * @brief Abstract class representing the functional model
     *
     * This class is an interface of the functional model. It offers the
     * functional method "F" which requires that the parameters of X be
     * in mathematical space. It contains normalization methods to transform
     * X from and to physical space. It also allows to retrieve the dimensions
     * of the problem.
     *
     */
    class FunctionalModel
    {
    public:
        /**
         * This method calculates y = F(x) using armadillo library and writes
         * results on the vector y without allocating new memory. This method
         * is used only by the other components of the kernel.
         *
         * @param x : vector of the functional model parameters. (L dimension)
         * @param y : vector of results (D dimension)
         */
        virtual void F(rowvec x, rowvec &y) = 0;

        rowvec F(rowvec x)
        {
            rowvec y(get_D_dimension());
            F(x, y);
            return y;
        }

        /**
         * This method returns the D dimension of the problem
         * @return the dimension D of the problem
         */
        virtual int get_D_dimension() = 0;

        /**
         * This method returns the L dimension of the problem
         * @return the dimension L of the problem
         */
        virtual int get_L_dimension() = 0;

        /**
         * This method transforms the values of x from the mathematical
         * space to the physical space.
         * @param x : the vector to normalize
         */
        virtual void to_physic(rowvec &x) = 0;

        /**
         * This method transforms the values of x from the mathematical
         * space to the physical space.
         * @param x : the vector to normalize
         */
        virtual void from_physic(rowvec &x) = 0;
    };

}

#endif // KERNELO_FUNCTIONNALMODEL_H
