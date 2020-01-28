/**
 * @file creators.h
 * @brief This file contains all data generation configuration structs responsible of the creation
 * of various statistic models. theses structs should be exported to fonrt-end language instead of
 * actual statsitci models.
 * @author Sami DJOUADI
 * @version 1.0
 * @date 27/01/2020
 */

#ifndef KERNELO_DATAGENCREATORS_H
#define KERNELO_DATAGENCREATORS_H

#include <string>
#include <memory>
#include "StatModel.h"
#include "GaussianStatModel.h"
#include "DependentGaussianStatModel.h"

namespace DataGeneration{

    struct GaussianStatModelConfig{
        std::string generatorType;
        double *covariance;
        int cov_size;
        unsigned seed;

        GaussianStatModelConfig(std::string generatorType, double *covariance, int cov_size, unsigned seed){
            this->generatorType = generatorType;
            this->covariance = covariance;
            this->cov_size = cov_size;
            this->seed = seed;
        }

        std::shared_ptr<StatModel> create(){
            return std::shared_ptr<StatModel>(
                    new GaussianStatModel(generatorType,covariance,cov_size,seed)
                    );
        }
    };

    struct DependentGaussianStatModelConfig{
        std::string generatorType;
        int r;
        unsigned seed;

        DependentGaussianStatModelConfig(std::string generatorType, int r, unsigned seed){
            this->generatorType = generatorType;
            this->r = r;
            this->seed = seed;
        }

        std::shared_ptr<StatModel> create(){
            return std::shared_ptr<StatModel>(
                    new DependentGaussianStatModel(generatorType,r,seed)
            );
        }
    };
}

#endif //KERNELO_DATAGENCREATORS_H
