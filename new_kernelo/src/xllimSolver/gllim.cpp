#include "gllim.hpp"
#include "../utils/utils.hpp"

// TODO

// GLLiM::GLLiM(unsigned D, unsigned L, unsigned K, GLLiMParameters &theta, GLLiMConstraints &constraints)
// {
// }

GLLiM::GLLiM(unsigned L, unsigned D, unsigned K) : theta(L, D, K), theta_star(D, L, K) // Initialize GLLiMParameters with the required arguments
{
    std::cout << "GLLiM Parameters initialized" << std::endl;
}

// void GLLiM::initialize(const mat &x, const mat &y, unsigned seed, unsigned nb_iter_EM, unsigned nb_experiences, unsigned max_iteration, double ratio_ll, double floor, unsigned kmeans_iteration, unsigned em_iteration, double floor)
// {
// }

// void GLLiM::train(const mat &x, const mat &y, unsigned max_iteration, double ratio_ll, double floor)
// {
// }

GLLiMParameters GLLiM::getParams()
{
    return this->theta;
}

std::string GLLiM::getDimensions()
{
    std::string str = "GLLiM dimensions are (L=" + std::to_string(this->theta.L) + ", D=" + std::to_string(this->theta.D) + ", K=" + std::to_string(this->theta.K) + ")";
    return str;
}

rowvec GLLiM::getParamPi()
{
    return this->theta.Pi;
}

cube GLLiM::getParamA()
{
    return this->theta.A;
}

mat GLLiM::getParamC()
{
    return this->theta.C;
}

cube GLLiM::getParamGamma()
{
    return this->theta.Gamma;
}

mat GLLiM::getParamB()
{
    return this->theta.B;
}

cube GLLiM::getParamSigma()
{
    return this->theta.Sigma;
}

void GLLiM::setParams(const GLLiMParameters &theta)
{
    this->theta = theta;
}

void GLLiM::setParamPi(const rowvec &Pi)
{
    this->theta.Pi = Pi;
}

void GLLiM::setParamA(const cube &A)
{
    this->theta.A = A;
}

void GLLiM::setParamC(const mat &C)
{
    this->theta.C = C;
}

void GLLiM::setParamGamma(const cube &Gamma)
{
    this->theta.Gamma = Gamma;
}

void GLLiM::setParamB(const mat &B)
{
    this->theta.B = B;
}

void GLLiM::setParamSigma(const cube &Sigma)
{
    this->theta.Sigma = Sigma;
}

GLLiMParameters GLLiM::getInverse()
{
    this->inverse();
    return this->theta_star;
}

void GLLiM::inverse()
{
    for (unsigned k = 0; k < this->theta.K; k++)
    {
        if (theta.Pi(k) != 0)
        {
            theta_star.Pi(k) = theta.Pi(k);
            mat sigma_inv = theta.Sigma.slice(k).i();
            mat gamma_inv = theta.Gamma.slice(k).i();
            theta_star.C.col(k) = theta.A.slice(k) * theta.C.col(k) + theta.B.col(k);
            theta_star.Gamma.slice(k) = theta.Sigma.slice(k) + theta.A.slice(k) * theta.Gamma.slice(k) * theta.A.slice(k).t();
            theta_star.Sigma.slice(k) = (gamma_inv + mat(theta.A.slice(k).t()) * sigma_inv * mat(theta.A.slice(k))).i();
            theta_star.A.slice(k) = theta_star.Sigma.slice(k) * mat(theta.A.slice(k).t()) * sigma_inv;
            theta_star.B.col(k) = theta_star.Sigma.slice(k) * vec(gamma_inv * vec(theta.C.col(k)) - mat(theta.A.slice(k).t()) * sigma_inv * vec(theta.B.col(k)));
        }
    }
}

// returns posterior mean estimates E[yn|xn;θ]
// TODO write formula from Delaforge 2014
std::tuple<mat, cube, cube> GLLiM::directDensities(const mat &x)
{
    unsigned N_obs = x.n_cols;
    x.print("x");
    // Compute weights
    gmm_full gmm;
    mat weights(N_obs, this->theta.K);
    gmm.set_params(this->theta.C, this->theta.Gamma, this->theta.Pi);

    // #pragma omp parallel for
    for (unsigned k = 0; k < this->theta.K; k++)
    {
        if (this->theta.Pi(k) == 0)
        {
            weights.col(k) = -datum::inf;
        }
        else
        {
            weights.col(k) = gmm.log_p(x, k).t();
        }
    }
    weights.each_col() -= utils::logSumExp(weights, 1);
    weights = exp(weights); // normalized real weights

    // means
    cube means(N_obs, this->theta.D, this->theta.K);
    for (unsigned k = 0; k < this->theta.K; ++k) {
        // Compute the means for each k
        // Each column of 'x' is multiplied by A.slice(k) and then B.col(k) is added
        means.slice(k) = (this->theta.A.slice(k) * x).t() + arma::repmat(this->theta.B.col(k).t(), N_obs, 1);
    }

    // covariances
    cube covariances = this->theta.Sigma; // The covariance is indenpendent from x

    return std::make_tuple(weights, means, covariances);
}

// Fast and general implementation of log-density. Works with Full covariance matrix
// void GLLiM::inverseDensities(const mat &y)
// {
//     gmm_full gmm;
//     gmm.set_params(this->theta_star.C, this->theta_star.Gamma, this->theta_star.Pi);
//     return gmm.log_p(y).t();

//     double log_det_gamma;
//     vec weights(gllim.K, fill::zeros);

//     for (unsigned k = 0; k < gllim.K; k++)
//     {
//         if (gllim.Pi(k) == 0)
//         {
//             weights(k) = -datum::inf;
//         }
//         else
//         {
//             mat Gamma_k = gllim.Gamma[k].getFull(); // je pense que cela prend du temps surtout lorsque D est grand.
//             weights(k) = log(gllim.Pi(k)) + Helpers::mvnrm_arma_fast_chol(rowvec(x.t()), rowvec(gllim.C.col(k).t()), Gamma_k);
//         }
//     }

//     return weights;
// }

// void GLLiM::getInsights()
// {
// }
