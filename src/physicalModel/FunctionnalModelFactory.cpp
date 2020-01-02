//
// Created by reverse-proxy on 29‏/12‏/2019.
//

#include "FunctionnalModelFactory.h"

#include <utility>
#include "Hapke02Model.h"
#include "Hapke93Model.h"


std::shared_ptr<FunctionnalModel> FunctionnalModelFactory::getModel(std::string type, std::vector<std::vector<double>> &data) {
    if(type == "hapke02"){
        return std::shared_ptr<FunctionnalModel> (new Hapke02Model(data));
    }else {
        return std::shared_ptr<FunctionnalModel> (new Hapke93Model(data));
    }
}
