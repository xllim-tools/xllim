#ifndef KERNELO_SHKURATOVMODEL_H
#define KERNELO_SHKURATOVMODEL_H

#include "FunctionalModel.hpp"

namespace Functional
{

    /**
     * @class ShkuratovModel
     * @brief A class describing the Shkuratov's model
     *
     * @details This class inherits @ref FunctionalModel "FunctionalModel" and overrides its methods by respecting the
     * equations in Shkuratov's model.
     *
     */
    class ShkuratovModel : public FunctionalModel
    {
    public:
        /**
         * @brief Constructor
         * @param geometries : pointer to the matrix of geometries that will be used by the model.
         * @param row_size : number of geometries.
         * @param col_size : the dimension of the geometries (should equals 3).
         * @param variant : The variant of the model corresponding to the number of parameters. ('5p' or '3p')
         * @param scalingCoeffs : A set of coefficients used in the transformation between physical and mathematical spaces.
         * @param offset : Offsets used in the transformation between physical and mathematical spaces.
         */
        ShkuratovModel(mat geometries, std::string variant, vec scalingCoeffs, vec offset);
        void F(vec photometry, vec &reflectances) final;
        int get_D_dimension() final;
        int get_L_dimension() final;
        void to_physic(vec &x) final;
        void from_physic(vec &x) final;

    protected:
        mat configuredGeometries; /** A matrix of the configured geometries */
        vec scalingCoeffs;        /** A set of coefficients used in the transformation between physical and mathematical spaces. */
        vec offset;               /** Offsets used in the transformation between physical and mathematical spaces */
        vec cos_i;                /** cos_i  directly computed from incidence angle */
        unsigned int L_dimension; /** The dimension corresponds the the model variant */

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

        static constexpr double DEGREE_180 = 180;
        enum photometry
        {
            AN = 0,
            MU_1 = 1,
            NU = 2,
            M = 3,
            MU_2 = 4
        };
        enum geometry
        {
            ALPHA = 0,
            BETA = 1,
            GAMMA = 2,
            INC = 0,
            EME = 1,
            PHI = 2
        };
    };

}

#endif // KERNELO_SHKURATOVMODEL_H
