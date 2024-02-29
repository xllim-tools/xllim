#ifndef KERNELO_DEPENDENTGAUSSIANSTATMODEL_H
#define KERNELO_DEPENDENTGAUSSIANSTATMODEL_H

#include "StatModel.hpp"
#include "../generator/Generator.hpp"
// #include <memory>

using namespace Functional;

namespace DataGeneration{

    /**
     * @class DependentGaussianStatModel
     * @brief A class representing a statistical model using the generated X matrix to add noise to Y.
     *
     * @details This class generates X matrix using a @ref Generator "data generator". This matrix is used
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
        DependentGaussianStatModel(
                const std::string& generatorType,
                std::shared_ptr<FunctionalModel> functionalModel,
                double r,
                unsigned int seed);

        std::tuple<mat, mat> gen_data(unsigned int n) final;

        // double density_X_Y(const vec &x, const vec &y, const vec &y_cov) final;

    private:
        std::shared_ptr<FunctionalModel> functionalModel;
        std::shared_ptr<Generator> generator; /**< generates X matrix values */
        double r; /**< SNR */
        unsigned seed; /**< used by generators */
    };
}



#endif //KERNELO_DEPENDENTGAUSSIANSTATMODEL_H
