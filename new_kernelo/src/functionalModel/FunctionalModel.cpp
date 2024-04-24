#include "FunctionalModel.hpp"
#include "../dataGeneration/generator/GeneratorFactory.hpp"
#include "../dataGeneration/generator/Generator.hpp"
// #include "../utils/utils.hpp"
#include <omp.h>

// TODO
std::tuple<mat, mat> FunctionalModel::genData(unsigned N, const std::string &generator_type, vec &covariance, unsigned seed)
{
    unsigned dimension_D = this->getDimensionY();
    unsigned dimension_L = this->getDimensionX();
    mat x_gen = mat(N, dimension_L);
    mat y_gen = mat(N, dimension_D);

    // generate X
    std::shared_ptr<DataGeneration::Generator> generator = DataGeneration::GeneratorFactory::create(generator_type, seed);
    generator->execute(x_gen);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normal_distribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    // generate Y
    vec noise(dimension_D);
    vec y_temp(dimension_D);

#pragma omp parallel for
    for (unsigned i = 0; i < N; i++)
    {
        // calculate F(X)
        this->F(x_gen.row(i).t(), y_temp);

        // add noise
        for (unsigned j = 0; j < dimension_D; j++)
        {
            noise(j) = normal_distribution(engine);
            y_gen(i, j) = y_temp(j) + noise(j) * covariance(j);
        }
    }

    return std::tuple<mat, mat>(x_gen, y_gen);
}

std::tuple<mat, mat> FunctionalModel::genData(unsigned N, const std::string &generator_type, double noise_ratio, unsigned seed)
{
    std::cout << "genData with double" << std::endl;
    unsigned dimension_D = this->getDimensionY();
    unsigned dimension_L = this->getDimensionX();
    mat x_gen = mat(N, dimension_L);
    mat y_gen = mat(N, dimension_D);

    // generate X
    std::shared_ptr<DataGeneration::Generator> generator = DataGeneration::GeneratorFactory::create(generator_type, seed);
    generator->execute(x_gen);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normal_distribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    // generate Y
    vec noise(dimension_D);
    vec y_temp(dimension_D);

#pragma omp parallel for
    for (unsigned i = 0; i < N; i++)
    {
        // calculate F(X)
        this->F(x_gen.row(i).t(), y_temp);

        // add noise
        for (unsigned j = 0; j < dimension_D; j++)
        {
            noise(j) = normal_distribution(engine);
            y_gen(i, j) = y_temp(j) + noise(j) * y_temp(j) / noise_ratio;
        }
    }

    return std::tuple<mat, mat>(x_gen, y_gen);
}

void FunctionalModel::targetDensity(vec &x, vec &y, vec &y_err, vec &noise, bool log)
{
}

void FunctionalModel::targetDensity(vec &x, vec &y, vec &y_err, double noise_ratio, bool log)
{
}

void FunctionalModel::importanceSampling(vec &weights, mat &means, cube &covariances, vec &y, vec &y_err, unsigned N_0, unsigned B, unsigned J)
{
}
