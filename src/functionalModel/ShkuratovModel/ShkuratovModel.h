/**
 * @file ShkuratovModel.h
 * @brief Shkuratov model class definition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 12/03/2020
 */


#ifndef KERNELO_SHKURATOVMODEL_H
#define KERNELO_SHKURATOVMODEL_H

#include "../Enumeration.h"
#include "../FunctionalModel.h"


namespace Functional {
    /**
     * @class ShkuratovModel
     * @brief A class describing the Shkuratov's model
     *
     * @details This class inherits @ref FunctionalModel "FunctionalModel" and overrides its methods by respecting the
     * equations in Shkuratov's model.
     *
     */
    class ShkuratovModel : public FunctionalModel {
    public:
        /**
         * @brief Constructor
         * @param geometries : pointer to the matrix of geometries that will be used by the model.
         * @param row_size : number of geometries.
         * @param col_size : the dimension of the geometries (should equals 3).
         * @param scalingCoeffs : A set of coefficients used in the transformation between physical and mathematical spaces.
         * @param offset : Offsets used in the transformation between physical and mathematical spaces.
         */
        ShkuratovModel(const double *geometries, int row_size, int col_size, const double *scalingCoeffs, const double *offset);
        void F(rowvec photometry, rowvec &reflectances) final;
        int get_D_dimension() final;
        int get_L_dimension() final;
        void to_physic(rowvec &x) final;
        void from_physic(double *x, int size) final;

    protected:
        mat configuredGeometries; /**< A matrix of the configured geometries */
        vec scalingCoeffs; /**< A set of coefficients used in the transformation between physical and
 * mathematical spaces. */
        vec offset; /**< Offsets used in the transformation between physical and
 * mathematical spaces. */

    private:
        /**
         * This method configures the geometries and prepares it for the calculation of reflectances
         * @param geometries : a matrix of 3 columns (theta, theta_0, psi)
         */
        void setupGeometries(const mat &geometries);

        /**
         * This method transforms a value in degree to a value in gradient.
         * @param degree
         * @return gradient value
         */
        static double degToGrad(double degree);
    };

}


#endif //KERNELO_SHKURATOVMODEL_H
