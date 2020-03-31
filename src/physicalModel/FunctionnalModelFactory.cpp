//
// Created by reverse-proxy on 29‏/12‏/2019.
//

#include "FunctionnalModelFactory.h"

#include <utility>
#include "HapkeModel/HapkeVersions/Hapke02Model.h"
#include "HapkeModel/HapkeVersions/Hapke93Model.h"


std::shared_ptr<FunctionalModel> FunctionnalModelFactory::getModel(std::string type, const double *data, int row_size, int col_size) {
    /*if(type == "hapke02"){
        return std::shared_ptr<FunctionalModel> (new Hapke02Model(data, row_size, col_size, 0, 0));
    }else {
        return std::shared_ptr<FunctionalModel> (new Hapke93Model(data, row_size, col_size, 0, 0));
    }*/
    return std::shared_ptr<FunctionalModel>();

}
