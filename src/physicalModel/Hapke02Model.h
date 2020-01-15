/**
 * @file Hapke02Model.h
 * @brief 2002 Hapke model class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 27/12/2019
 */
#ifndef UNTITLED_HAPKE02MODEL_H
#define UNTITLED_HAPKE02MODEL_H

#include "HapkeModel.h"

/**
 * @class Hapke02Model
 * @brief Abstract class representing the 2002 version of the Hapke model
 *
 * @details This class overrides the uncommon parts of the reflectance formula.
 *
 */
class Hapke02Model : public HapkeModel {
public:
    /**
     * @brief Constructor
     * @details Hapke02Model class constructor
     * @param geometries : matrix of geometries that will be used by the model
     */
    Hapke02Model(const double *geometries, int row_size, int col_size, const std::shared_ptr<HapkeAdapter>& adapter);

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



#endif //UNTITLED_HAPKE02MODEL_H
