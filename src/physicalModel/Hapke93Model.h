/**
 * @file Hapke93Model.h
 * @brief 1993 Hapke model class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 25/12/2019
 */

#ifndef KERNELO_HAPKE93MODEL_H
#define KERNELO_HAPKE93MODEL_H

#include "HapkeModel.h"

namespace Functional {
/**
 * @class Hapke93Model
 * @brief A class representing the 1993 version of Hapke's model
 *
 * @details This class overrides the varying parts of the reflectance formula.
 *
 * See : Hapke B. 1993 Theory of Reflectance and Emittance Spectroscopy. Topics in Remote Sensing.
 * Cambridge University Press, Cambridge, UK.
 *
 * See : Schmidt F. and Fernando J. 2015 Realistic uncertainties on Hapke model parameters from
 * photometric measurement. Icarus, 260 :73 - 93, 2015.
 * 
 */
    class Hapke93Model : public HapkeModel {
    public:
        /**
         * @brief Constructor
         * @details Hapke93Model class constructor
         * @param geometries : matrix of geometries that will be used by the model
         * @param row_size : number of geometries.
         * @param col_size : number of parameters per geometry (Dimenion D = 3).
         * @param adapter : a shared pointer to the @ref HapkeAdapter "adapter".
         */
        Hapke93Model(const double *geometries, int row_size, int col_size,
                     const std::shared_ptr<HapkeAdapter> &adapter);

    private:
        double set_coef() override;

        rowvec define_different_part(const rowvec &photometry, rowvec mue, rowvec mu0e) override;

        /**
         * This method calculates the multiple scattering
         * @param x
         * @param omega : single scattering albedo
         * @return a vector of D results
         */
        static rowvec calculate_H(const rowvec &x, double omega);
    };
}

#endif //KERNELO_HAPKE93MODEL_H
