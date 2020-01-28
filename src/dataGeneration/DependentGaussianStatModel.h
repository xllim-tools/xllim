/**
 * @file DependentGaussianStatModel.h
 * @brief Class definition of a gaussian statistical model where noise depends on generated X.
 * @author Sami DJOUADI
 * @version 1.0
 * @date 17/01/2020
 */

#ifndef KERNELO_DEPENDENTGAUSSIANSTATMODEL_H
#define KERNELO_DEPENDENTGAUSSIANSTATMODEL_H

#include "StatModel.h"
#include "GeneratorStrategy.h"
#include <memory>

using namespace Functional;

namespace DataGeneration{

    /**
     * @class DependentGaussianStatModel
     * @brief A class representing a statistical model using the generated X matrix to add noise to Y.
     *
     * @details This class generates X matrix using a @ref GeneratorStrategy "data generator". This matrix is used
     * to calculate Y through the functional model to which an X dependent noise is added. The class requires the type
     * of the data generator to use, the noise effect in percentage and a seed used by generators.
     */
    class DependentGaussianStatModel : public StatModel{
    public:
        /**
         * @brief Constructor
         * @details DependentGaussianStatModel class constructor
         * @param generatorType : the ype of the generator used to generate X matrix values
         * @param r : noise effect in percentage ( example 20 means 20%)
         * @param seed : used by generators
         */
        DependentGaussianStatModel(std::string generatorType, int r, unsigned seed);
        std::tuple<mat, mat> gen_data(std::shared_ptr<FunctionalModel> functionalModel, int n) final;
        double density_X_Y(mat x, mat y) final;

    private:
        std::shared_ptr<GeneratorStrategy> generator; /**< generates X matrix values */
        double r; /**< noise effect in percentage */
        unsigned seed; /**< used by generators */
    };
}



#endif //KERNELO_DEPENDENTGAUSSIANSTATMODEL_H
