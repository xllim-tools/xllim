#include "GaussianStatModel.hpp"
#include "../generator/GeneratorFactory.hpp"
#include "../../../../src/helpersFunctions/Helpers.h"
#include <omp.h>

#define LOG_2_PI log(2 * datum::pi)

using namespace std;
using namespace DataGeneration;

GaussianStatModel::GaussianStatModel(
    const std::string &generatorType,
    std::shared_ptr<FunctionalModel> functionalModel,
    vec covariance,
    unsigned int seed)
{
    this->generator = GeneratorFactory::create(generatorType, seed);
    this->functionalModel = std::move(functionalModel);
    this->covariance = covariance;
    this->seed = seed;
}

std::tuple<mat, mat> GaussianStatModel::gen_data(unsigned int n)
{
    unsigned int dimension_D = functionalModel->getDimensionY();
    unsigned int dimension_L = functionalModel->getDimensionX();

    mat x_arma = mat(n, dimension_L);
    mat y_arma = mat(n, dimension_D);

    // generate X
    generator->execute(x_arma);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normalDistribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    // generate Y
    vec noise(dimension_D);
    vec y_temp(dimension_D);

#pragma omp parallel for
    for (unsigned i = 0; i < n; i++)
    {
        // calculate F(X)
        functionalModel->F(x_arma.row(i).t(), y_temp);

        // add noise
        for (unsigned j = 0; j < dimension_D; j++)
        {
            noise(j) = normalDistribution(engine);
            y_arma(i, j) = y_temp(j) + noise(j) * covariance(j);
        }
    }

    return std::tuple<mat, mat>(x_arma, y_arma);
}

// double GaussianStatModel::density_X_Y(const vec &x, const vec &y, const vec &y_cov)
// {
//     for (auto x_i : x)
//     {
//         if (x_i > 1 || x_i < 0)
//         {
//             return -datum::inf;
//         }
//     }
//     vec y_u(y.n_rows);
//     this->functionalModel->F(x, y_u);
//     y_u = y.t() - y_u;

//     mat cov = mat(this->functionalModel->getDimensionY(), this->functionalModel->getDimensionY(), fill::zeros);
//     cov.diag() += pow(y_cov, 2) + pow(covariance.t(), 2);
//     //    std::cout << "det :" << Helpers::computeDeterminant(cov) << std::endl;
//     //    std::cout << "dot :" << dot(y_u.t() % (1 / (pow(y_cov ,2)+ pow(covariance.t(),2))), y_u.t()) << std::endl;

//     return -0.5 * (y_cov.n_rows * LOG_2_PI +
//                    Helpers::computeDeterminant(cov) +
//                    dot(y_u.t() % (1 / (pow(y_cov, 2) + pow(covariance.t(), 2))), y_u.t()));
// }
