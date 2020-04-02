//
// Created by reverse-proxy on 2‏/4‏/2020.
//

#ifndef KERNELO_ISTARGET_H
#define KERNELO_ISTARGET_H

#include <memory>
#include <utility>
#include "../dataGeneration/StatModel.h"

namespace importanceSampling{
    struct ISTarget{
        std::shared_ptr<DataGeneration::StatModel> target;
    };
}

#endif //KERNELO_ISTARGET_H
