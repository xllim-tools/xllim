/**
 * @file SixParamsModel.h
 * @brief Class definition of the 6 parameters Hapke model adapter
 * @author Sami DJOUADI
 * @version 1.0
 * @date 14/12/2019
 */

#ifndef KERNELO_SIXPARAMSMODEL_H
#define KERNELO_SIXPARAMSMODEL_H

#include "HapkeAdapter.h"
namespace Functional{
    /**
     * @class SixParamsModel
     * @brief A class that computes the whole Hapke's model with 6 parameters
     */
    class SixParamsModel: public HapkeAdapter {
    public:
        /**
         * Default constructor
         */
        SixParamsModel();

        void adaptModel(rowvec &photometry) override ;
        int get_dimension_L() override ;


    };
}


#endif //KERNELO_SIXPARAMSMODEL_H
