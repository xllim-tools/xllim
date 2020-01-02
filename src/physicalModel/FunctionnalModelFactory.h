//
// Created by reverse-proxy on 29‏/12‏/2019.
//

#ifndef UNTITLED_FUNCTIONNALMODELFACTORY_H
#define UNTITLED_FUNCTIONNALMODELFACTORY_H

#include "FunctionnalModel.h"

class FunctionnalModelFactory {
public:
    static std::shared_ptr<FunctionnalModel> getModel(std::string type, std::vector<std::vector<double>> &data);
};


#endif //UNTITLED_FUNCTIONNALMODELFACTORY_H
