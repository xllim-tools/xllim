//
// Created by reverse-proxy on 27‏/1‏/2020.
//

#ifndef KERNELO_DATAGENCREATORS_H
#define KERNELO_CREATORS_H

#include <string>
#include <memory>
#include "HapkeAdapter.h"
#include "HapkeAdapterFactory.h"
#include "Hapke02Model.h"
#include "SixParamsModel.h"
#include "FourParamsModel.h"
#include "ThreeParamsModel.h"
#include "Hapke93Model.h"

namespace Functional{
    struct HapkeAdapterConfig{
        std::string version;
        double b0;
        double h;

        std::shared_ptr<HapkeAdapter> create(){
            if(version == "six") {
                return std::shared_ptr<HapkeAdapter>(new SixParamsModel());
            }else if(version == "four"){
                return std::shared_ptr<HapkeAdapter>(new FourParamsModel(this->b0, this->h));
            }else if(version == "three"){
                return std::shared_ptr<HapkeAdapter>(new ThreeParamsModel(this->b0, this->h));
            }
        }
    };

    struct HapkeModelConfig{
        std::string version;
        HapkeAdapterConfig adapterConfig;
        const double *geometries;
        int row_size;
        int col_size;
        double theta_bar_scalling;

        std::shared_ptr<FunctionalModel> create(){
            std::shared_ptr<HapkeAdapter> adapter = this->adapterConfig.create();
            if(version == "2002"){
                std::cout << &geometries[0] << std::endl;
                return std::shared_ptr<FunctionalModel>(
                        new Hapke02Model(
                                geometries,
                                row_size,
                                col_size,
                                adapter,
                                theta_bar_scalling)
                                );
            }
            else if(version == "1993"){
                return std::shared_ptr<FunctionalModel>(
                        new Hapke93Model(
                                geometries,
                                row_size,
                                col_size,
                                adapter,
                                theta_bar_scalling)
                                );
            }
        }
    };
}

#endif //KERNELO_DATAGENCREATORS_H
