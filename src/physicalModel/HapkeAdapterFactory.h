//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#ifndef KERNELO_HAPKEADAPTERFACTORY_H
#define KERNELO_HAPKEADAPTERFACTORY_H

#include <memory>
#include "HapkeAdapter.h"

using namespace Functional;

class HapkeAdapterFactory {
public:
    static std::shared_ptr<HapkeAdapter> create(int dimension_L);
};


#endif //KERNELO_HAPKEADAPTERFACTORY_H
