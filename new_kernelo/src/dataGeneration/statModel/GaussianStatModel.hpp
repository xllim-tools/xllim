#ifndef KERNELO_GAUSSSTATMODEL_H
#define KERNELO_GAUSSSTATMODEL_H

#include "StatModel.hpp"
#include "../generator/Generator.hpp"
#include <memory>

using namespace Functional;


namespace DataGeneration{

    /**
     * @class GaussianStatModel
     * @brief A class representing a statistical model using a normal distribution to create a noise in data
     * generation process.
     *
     * @details This class generates X matrix using a @ref Generator "data generator". This matrix is used
     * to calculate Y through the functional model to which a gaussian noise is added. The class requires the type
     * of the data generator to use , the matrix of the covariance and a seed used by generators.
     */
    class GaussianStatModel : public StatModel{
    public :
        /**
         * @brief Constructor
         * @details GaussianStatModel class constructor
         * @param generatorType : the ype of the generator used to generate X matrix values
         * @param covariance : covariance matrix used to add noise to Y
         * @param cov_size : covariance square matrix size
         * @param seed : used by generators
         */
        GaussianStatModel(
                const std::string &generatorType,
                std::shared_ptr<FunctionalModel> functionalModel,
                vec covariance,
                unsigned int seed);

        std::tuple<mat, mat> gen_data(unsigned int n) final;

        // double density_X_Y(const vec &x, const vec &y, const vec &y_cov) final;

    private:
        std::shared_ptr<FunctionalModel> functionalModel;
        std::shared_ptr<Generator> generator; /**< generates X matrix values */
        vec covariance; /**< covariance row vector used to add noise to Y */
        unsigned seed; /**< used by generators */
    };

}



#endif //KERNELO_GAUSSSTATMODEL_H
