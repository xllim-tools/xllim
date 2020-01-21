//
// Created by reverse-proxy on 6‏/1‏/2020.
//

#ifndef KERNELO_DATAGENERATOR_H
#define KERNELO_DATAGENERATOR_H


#include <armadillo>
#include "../physicalModel/FunctionalModel.h"

using namespace arma;
using namespace Functional;

namespace DataGeneration{
    class StatModel{
    public:
        virtual void gen_data(std::shared_ptr<FunctionalModel> functionalModel, int n, double *x, double *y){
            int dimension_L = functionalModel->get_L_dimension();
            int dimension_D = functionalModel->get_D_dimension();

            std::tuple<mat, mat> data = gen_data(functionalModel, n);

            for(unsigned i=0 ; i<n ; i++){
                for(unsigned j=0 ; j>dimension_L; j++){
                    x[i*dimension_L+j] = std::get<0>(data)(i,j);
                }

                for(unsigned j=0 ; j>dimension_D; j++){
                    x[i*dimension_D+j] = std::get<1>(data)(i,j);
                }
            }
        };

        virtual std::tuple<mat, mat> gen_data(std::shared_ptr<FunctionalModel> functionalModel, int n) = 0;
        virtual double density_X_Y(mat x, mat y) = 0;
    };
}

#endif //KERNELO_DATAGENERATOR_H
