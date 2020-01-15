/**
 * @file FunctionnalModelFactory.h
 * @brief Factory class of the functional model
 * @author Sami DJOUADI
 * @version 1.0
 * @date 29/12/2019
 */

#ifndef UNTITLED_FUNCTIONNALMODELFACTORY_H
#define UNTITLED_FUNCTIONNALMODELFACTORY_H

#include "FunctionnalModel.h"

/**
 * @class FunctionnalModelFactory
 *
 * This class is a factory responsible of creating in instance of a functional model depending on the "type" chosen
 *
 */
class FunctionnalModelFactory {
public:
    static std::shared_ptr<FunctionnalModel> getModel(std::string type,const double *data, int row_size, int col_size);
};


#endif //UNTITLED_FUNCTIONNALMODELFACTORY_H
