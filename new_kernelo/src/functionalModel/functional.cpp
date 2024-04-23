#include "functional.hpp"

// TODO

void FunctionalModel::genData(unsigned N, std::string &generator_type, vec &noise, unsigned seed)
{
}

void FunctionalModel::genData(unsigned N, std::string &generator_type, double noise_ratio, unsigned seed)
{
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


