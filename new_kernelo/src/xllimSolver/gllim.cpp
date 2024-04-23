#include "gllim.hpp"

// TODO

GLLiM::GLLiM(unsigned D, unsigned L, unsigned K, GLLiMParameters &theta, GLLiMConstraints &constraints)
{
}

void GLLiM::initialize(const mat &x, const mat &y, unsigned seed, unsigned nb_iter_EM, unsigned nb_experiences, unsigned max_iteration, double ratio_ll, double floor, unsigned kmeans_iteration, unsigned em_iteration, double floor)
{
}

void GLLiM::train(const mat &x, const mat &y, unsigned max_iteration, double ratio_ll, double floor)
{
}

GLLiMParameters GLLiM::getParams()
{
    return GLLiMParameters();
}

GLLiMParameters GLLiM::getParamA()
{
    return GLLiMParameters();
}

void GLLiM::setParams(GLLiMParameters &theta)
{
}

void GLLiM::setParamA(cube A)
{
}

GLLiMParameters GLLiM::getInverse()
{
    return GLLiMParameters();
}

void GLLiM::directDensities(const mat &x)
{
}

void GLLiM::inverseDensities(const mat &y)
{
}

void GLLiM::getInsights()
{
}
