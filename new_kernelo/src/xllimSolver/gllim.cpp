#include "gllim.hpp"
#include "../utils/utils.hpp"
#include "jgmm.hpp"

// TODO

// GLLiM::GLLiM(unsigned D, unsigned L, unsigned K, GLLiMParameters &theta, GLLiMConstraints &constraints)
// {
// }

// TODO est ce qu'on veut que les contraintes (full, diag..) soient définies par la classe ?
// TODO Si oui alors il faut que les cstr soient def à l'initialisation et il faut que les checks de Params prennent en compte les contraintes.
// TODO     Dans ce cas peut-être qu'il faudrait faire des template ou des classes dérivées (FullGllim,...)
// TODO Sinon on doit faire un check des matrices avant les calculs. On n'a pas de arma::is_isotropic(). Voir comment ça s'articule dans les calculs.
// TODO En soit on a juste besoin de savoir les propriétées des matrices pour certains calculs. Mais dans la théorie, il est intéressant de savoir si on est en Full, Diag ou Iso ou fixed?
// TODO utilisation mémoire à optimiser en cas de Diag ou iso ...
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
void GLLiM::train(const mat &x, const mat &y, unsigned kmeans_iteration, unsigned em_iteration, double floor)
{
    // this->checkConstraints(); // ? Check if Params are valid and update constraints

    // Full/Diag case
    // if (this->sigma_cstr_type == 'full' && this->gamma_cstr_type == 'full')
    // if (this->theta.Sigma.is_diagmat() || this->theta.Gamma.is_diagmat())
    // {
    //     // TODO
    // }
    // else
    // {
    // GLLiM is equivalent to a classic GMM on the joint law (X,Y). Applying the Armadillo built-in EM method is more efficient.
    JGMM estimator;
    this->theta = estimator.train(x, y, this->theta, kmeans_iteration, em_iteration, floor); //  comment faire avec les paramètres ?
    // }
}

// void GLLiM::checkConstraints()
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
    std::string err = "";
    if (!(theta.Pi.is_vec() && (theta.Pi.n_cols == this->theta.K)))
    {
        err += "Pi dimensions must be of shape (" + std::to_string(this->theta.K) + ")\n";
    }
    if (!(abs(accu(theta.Pi) - 1.0) < 1e-9))
    {
        err += "The sum of weights must be equal to 1\n";
    }
    if (!(arma::size(theta.A) == arma::SizeCube(this->theta.D, this->theta.L, this->theta.K)))
    {
        err += "A dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.L) + "," + std::to_string(this->theta.K) + ")\n";
    }
    if (!(arma::size(theta.C) == arma::SizeMat(this->theta.L, this->theta.K)))
    {
        err += "C dimensions must be of shape (" + std::to_string(this->theta.L) + "," + std::to_string(this->theta.K) + ")\n";
    }
    if (!(arma::size(theta.Gamma) == arma::SizeCube(this->theta.L, this->theta.L, this->theta.K)))
    {
        err += "Gamma dimensions must be of shape (" + std::to_string(this->theta.L) + "," + std::to_string(this->theta.L) + "," + std::to_string(this->theta.K) + ")\n";
    }
    if (!(arma::size(theta.B) == arma::SizeMat(this->theta.D, this->theta.K)))
    {
        err += "B dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.K) + ")\n";
    }
    if (!(arma::size(theta.Sigma) == arma::SizeCube(this->theta.D, this->theta.D, this->theta.K)))
    {
        err += "Sigma dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.D) + "," + std::to_string(this->theta.K) + ")\n";
    }
    if (err == "")
    {
        this->theta = theta;
    }
    else
    {
        throw std::invalid_argument(err);
    }
}

void GLLiM::setParamPi(const rowvec &Pi)
{
    if (Pi.is_vec() && (Pi.n_cols == this->theta.K))
    {
        if ((abs(accu(Pi) - 1.0) < 1e-9))
        {
            this->theta.Pi = Pi;
        }
        else
        {
            throw std::invalid_argument("The sum of weights must be equal to 1.");
        }
    }
    else
    {
        throw std::invalid_argument("Pi dimensions must be of shape (" + std::to_string(this->theta.K) + ")");
    }
}

void GLLiM::setParamA(const cube &A)
{
    if (arma::size(A) == arma::SizeCube(this->theta.D, this->theta.L, this->theta.K))
    {
        this->theta.A = A;
    }
    else
    {
        throw std::invalid_argument("A dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.L) + "," + std::to_string(this->theta.K) + ")");
    }
}

void GLLiM::setParamC(const mat &C)
{
    if (arma::size(C) == arma::SizeMat(this->theta.L, this->theta.K))
    {
        this->theta.C = C;
    }
    else
    {
        throw std::invalid_argument("C dimensions must be of shape (" + std::to_string(this->theta.L) + "," + std::to_string(this->theta.K) + ")");
    }
}

void GLLiM::setParamGamma(const cube &Gamma)
{
    if (arma::size(Gamma) == arma::SizeCube(this->theta.L, this->theta.L, this->theta.K))
    {
        this->theta.Gamma = Gamma;
    }
    else
    {
        throw std::invalid_argument("Gamma dimensions must be of shape (" + std::to_string(this->theta.L) + "," + std::to_string(this->theta.L) + "," + std::to_string(this->theta.K) + ")");
    }
}

void GLLiM::setParamB(const mat &B)
{
    if (arma::size(B) == arma::SizeMat(this->theta.D, this->theta.K))
    {
        this->theta.B = B;
    }
    else
    {
        throw std::invalid_argument("B dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.K) + ")");
    }
}

void GLLiM::setParamSigma(const cube &Sigma)
{
    if (arma::size(Sigma) == arma::SizeCube(this->theta.D, this->theta.D, this->theta.K))
    {
        this->theta.Sigma = Sigma;
    }
    else
    {
        throw std::invalid_argument("Sigma dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.D) + "," + std::to_string(this->theta.K) + ")");
    }
}

GLLiMParameters GLLiM::getInverse()
{
    this->theta_star = this->inverse(this->theta);
    return this->theta_star;
}
// TODO Is the theta_star attribute really useful if it is recalculated every time and is differring from theta_star_altered

GLLiMParameters GLLiM::inverse(GLLiMParameters &theta)
{
    GLLiMParameters theta_star(theta.D, theta.L, theta.K);
    for (unsigned k = 0; k < theta.K; k++)
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
    return theta_star;
}

std::tuple<mat, cube, cube> GLLiM::constructGMM(const mat &x, GLLiMParameters &theta)
{
    unsigned N_obs = x.n_cols;

    // Compute weights
    mat weights(N_obs, theta.K);
    gmm_full gmm;

    // Full/Diag case
    // if (this->sigma_cstr_type == 'diag' || 'iso')
    // {
    //     gmm_full gmm;
    // }
    // else
    // {
    //     gmm_diag gmm;
    // }

    gmm.set_params(theta.C, theta.Gamma, theta.Pi);

#pragma omp parallel for
    for (unsigned k = 0; k < theta.K; k++)
    {
        if (theta.Pi(k) == 0)
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
    cube means(N_obs, theta.D, theta.K);
    for (unsigned k = 0; k < theta.K; ++k)
    {
        // Compute the means for each k
        // Each column of 'x' is multiplied by A.slice(k) and then B.col(k) is added
        means.slice(k) = (theta.A.slice(k) * x).t() + arma::repmat(theta.B.col(k).t(), N_obs, 1);
    }

    // covariances
    cube covariances = theta.Sigma; // The covariance is indenpendent from x

    return std::make_tuple(weights, means, covariances);
}

// returns posterior mean estimates E[yn|xn;θ]
// TODO write formula from Delaforge 2014
PredictionResult GLLiM::directDensities(const mat &x, const vec &x_incertitude)
{
    unsigned N_obs = x.n_cols;
    PredictionResult result(N_obs, this->theta.D, this->theta.K);

    // ==================== Alter theta covariance and inverse theta ====================

    GLLiMParameters theta_altered = this->theta;
    theta_altered.Gamma.each_slice() += diagmat(pow(x_incertitude, 2));

    // ==================== Construct the GMM of the forward conditional model ====================

    std::tuple<mat, cube, cube> GMMs = constructGMM(x, theta_altered);
    result.meanPredResult.gmm_weights = std::get<0>(GMMs); // (N_obs, K)
    result.meanPredResult.gmm_means = std::get<1>(GMMs);   // (N_obs, D, K)
    result.meanPredResult.gmm_covs = std::get<2>(GMMs);    // (D, D, K)

    // ============================= Prediction mean estimations ==================================

    // Compute the mean of the means in the mixture
    result.meanPredResult.mean = mat(N_obs, theta_altered.D);
    for (unsigned k = 0; k < theta_altered.K; ++k)
    {
        result.meanPredResult.mean += diagmat(result.meanPredResult.gmm_weights.col(k)) * result.meanPredResult.gmm_means.slice(k);
    }

// Compute the mean of covariances in the mixture
// TODO voir formule dans Kugler 2021. Can be simplified
#pragma omp parallel
    result.meanPredResult.variance = cube(N_obs, theta_altered.D, theta_altered.D);
    for (unsigned n = 0; n < N_obs; ++n)
    {
        for (unsigned k = 0; k < theta_altered.K; ++k)
        {
            rowvec mean_diff = result.meanPredResult.gmm_means.slice(k).row(n) - result.meanPredResult.mean.row(n);
            result.meanPredResult.variance.row(n) += result.meanPredResult.gmm_weights(n, k) * (result.meanPredResult.gmm_covs.slice(k) + mean_diff.t() * mean_diff);
        }
    }

    return result;
}

// returns prior mean estimates E[xn|yn;θ]
PredictionResult GLLiM::inverseDensitiesOneInversion(const mat &y, const vec &y_incertitude)
// TODO merge this method with directDensities. Check out the differences
{
    unsigned N_obs = y.n_cols;
    PredictionResult result(N_obs, this->theta.L, this->theta.K);

    // ==================== Alter theta covariance and inverse theta ====================

    GLLiMParameters theta_altered = this->theta;
    theta_altered.Sigma.each_slice() += diagmat(pow(y_incertitude, 2));
    GLLiMParameters theta_star_altered = inverse(theta_altered);

    // ==================== Construct the GMM of the forward conditional model ====================

    std::tuple<mat, cube, cube> GMMs = constructGMM(y, theta_star_altered);
    result.meanPredResult.gmm_weights = std::get<0>(GMMs); // (N_obs, K)
    result.meanPredResult.gmm_means = std::get<1>(GMMs);   // (N_obs, D, K)
    result.meanPredResult.gmm_covs = std::get<2>(GMMs);    // (D, D, K)

    // ============================= Prediction mean estimations ==================================

    // Compute the mean of the means in the mixture
    result.meanPredResult.mean = mat(N_obs, theta_star_altered.D); // theta_star_altered.D or theta_altered.L (the second one is more explicit. )
    for (unsigned k = 0; k < theta_star_altered.K; ++k)
    {
        result.meanPredResult.mean += diagmat(result.meanPredResult.gmm_weights.col(k)) * result.meanPredResult.gmm_means.slice(k);
    }

    // Compute the mean of covariances in the mixture
    // TODO voir formule dans Kugler 2021.  can be simplified
    result.meanPredResult.variance = cube(N_obs, theta_star_altered.D, theta_star_altered.D);
#pragma omp parallel
    for (unsigned n = 0; n < N_obs; ++n)
    {
        for (unsigned k = 0; k < theta_star_altered.K; ++k)
        {
            rowvec mean_diff = result.meanPredResult.gmm_means.slice(k).row(n) - result.meanPredResult.mean.row(n);
            result.meanPredResult.variance.row(n) += result.meanPredResult.gmm_weights(n, k) * (result.meanPredResult.gmm_covs.slice(k) + mean_diff.t() * mean_diff);
        }
    }

    return result;
}

// returns prior mean estimates E[xn|yn;θ] when the observation incertitude is different for each observation (no parallelisation)
PredictionResult GLLiM::inverseDensities(const mat &y, const mat &y_incertitude)
// TODO merge this method with directDensities. Check out the differences
{
    unsigned N_obs = y.n_cols;
    PredictionResult result(N_obs, this->theta.L, this->theta.K);

    if (y_incertitude.n_rows != y.n_rows)
    {
        throw std::invalid_argument("Observations and associated incertitudes have not the same dimension.");
    }
    else if (y_incertitude.n_cols != 1 && y_incertitude.n_cols != y.n_cols)
    {
        throw std::invalid_argument("The number of observations and associated incertitudes do not match.");
    }
    else if (y_incertitude.n_cols == 1)
    {
        result = GLLiM::inverseDensitiesOneInversion(y, vec(y_incertitude));
    }
    else
    {
#pragma omp parallel for
        for (size_t n = 0; n < N_obs; n++)
        {
            PredictionResult res_n = GLLiM::inverseDensitiesOneInversion(mat(y.col(n)), vec(y_incertitude.col(n)));
            result.meanPredResult.gmm_weights.row(n) = res_n.meanPredResult.gmm_weights; // (N_obs, K)
            result.meanPredResult.gmm_means.row(n) = res_n.meanPredResult.gmm_means;     // (N_obs, D, K)
            result.meanPredResult.gmm_covs = res_n.meanPredResult.gmm_covs;              // (D, D, K) // TODO Problem because for this case gmm_covs = theta_star.Sigma and is different for each observation :/
            result.meanPredResult.mean.row(n) = res_n.meanPredResult.mean;               // (N_obs, D)
            result.meanPredResult.variance.row(n) = res_n.meanPredResult.variance;       // (N_obs, D, D)
        }
    }

    return result;
}

// void GLLiM::getInsights()
// {
// }
