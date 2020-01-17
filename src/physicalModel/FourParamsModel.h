/**
 * @file FourParamsModel.h
 * @brief Class definition of the 4 parameters Hapke model adapter
 * @author Sami DJOUADI
 * @version 1.0
 * @date 14/12/2019
 */

#ifndef KERNELO_FOURPARAMSMODEL_H
#define KERNELO_FOURPARAMSMODEL_H

#include "HapkeAdapter.h"

namespace Functional{
    /**
     * @class FourParamsModel
     * @brief This class adapts @ref HapkeModel "hapke model" to models with 4 parameters {omega, theta_bar, b , c}.
     * By using a default or chosen values of b0 and h.
     */
    class FourParamsModel : public HapkeAdapter{
    public:
        /**
         * The constructor initializes b0 and h.
         * @param b0
         * @param h
         */
        FourParamsModel(double b0, double h);
        void adaptModel(rowvec &photometry) override ;
        int get_dimension_L() override ;
    };
}


#endif //KERNELO_FOURPARAMSMODEL_H
