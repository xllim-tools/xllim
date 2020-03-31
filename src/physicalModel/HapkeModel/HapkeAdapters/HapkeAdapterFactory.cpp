//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#include "HapkeAdapterFactory.h"
#include "SixParamsModel.h"
#include "ThreeParamsModel.h"

using namespace Functional;

std::shared_ptr<HapkeAdapter> HapkeAdapterFactory::create(int dimension_L) {

    /*if(dimension_L == 6){
        return std::shared_ptr<HapkeAdapter>(new SixParamsModel());
    }else if (dimension_L == 3){
        return std::shared_ptr<HapkeAdapter>(new ThreeParamsModel());
    }*/

    return nullptr;
}
