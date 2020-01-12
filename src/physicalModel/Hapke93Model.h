/**
 * @file Hapke93Model.h
 * @brief 1993 Hapke model class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 25/12/2019
 */

#ifndef UNTITLED_HAPKE93MODEL_H
#define UNTITLED_HAPKE93MODEL_H

#include "HapkeModel.h"

/**
 * @class Hapke93Model
 * @brief Abstract class representing the 1993 version of the Hapke model
 *
 * @details This class overrides the uncommon parts of the reflectance formula.
 *
 */
class Hapke93Model : public HapkeModel {
public:
    /**
     * @brief Constructor
     * @details Hapke93Model class constructor
     * @param geometries : matrix of geometries that will be used by the model
     */
    Hapke93Model(const double *geometries, int row_size, int col_size);

private:
    double set_coef() override ;
    rowvec define_different_part(const rowvec &x, rowvec mue, rowvec mu0e) override ;

    /**
     * This method calculates the multiple scattering
     * @param x
     * @param omega
     * @return a vector of D results
     */
    static rowvec calculate_H(const rowvec &x , double omega);
};


#endif //UNTITLED_HAPKE93MODEL_H
