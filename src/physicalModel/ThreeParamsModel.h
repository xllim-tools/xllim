/**
 * @file ThreeParamsModel.h
 * @brief Class implementation of the 3 parameters Hapke model adapter
 * @author Sami DJOUADI
 * @version 1.0
 * @date 14/12/2019
 *
 * See Hapke B. 2012, Bidirectional reflectance spectroscopy 7: The single particle phase function hockey
 * stick relation. Icarus 221 (2), 1079-1083.
 */

#ifndef KERNELO_THREEPARAMSMODEL_H
#define KERNELO_THREEPARAMSMODEL_H

#include "HapkeAdapter.h"

namespace Functional {
/**
 * @class ThreeParamsModel
 * @brief This class adapts @ref HapkeModel "hapke model" to models with 3 parameters {omega, theta_bar, b}.
 * By using a default or chosen values of b0 and h, and calculating c using the hokey stick relation.
 */
    class ThreeParamsModel : public HapkeAdapter {
    public:
        /**
         * The constructor initializes b0 and h,
         * @param b0
         * @param h
         */
        ThreeParamsModel(double b0, double h);

        void adaptModel(rowvec &photometry) override;

        int get_dimension_L() override;
    };
}


#endif //KERNELO_THREEPARAMSMODEL_H
