#include "gllim.hpp"
#include "../utils/utils.hpp"
#include "jgmm.hpp"
#include "emEstimator.hpp"

// ============================== Static methods ==============================

template <typename TCov>
static typename TCov::Type convertVectorOfCovToArma(unsigned K, unsigned dimension, std::vector<TCov> cov_vector)
{
    typename TCov::Type cov_arma(TCov::getTypeSize(K, dimension));
    for (unsigned k = 0; k < K; k++)
    {
        cov_arma.row(k) = cov_vector[k].get();
    }
    return cov_arma;
}

template <typename TCov>
static std::vector<TCov> convertArmaToVectorOfCov(unsigned &K, unsigned &dimension, const typename TCov::Type &cov_arma)
{
    std::vector<TCov> cov_vector(K, TCov(dimension));
    for (unsigned k = 0; k < K; k++)
    {
        cov_vector[k] = TCov(cov_arma.row(k));
    }
    return cov_vector;
}

// ============================== Constructors ==============================

// GLLiM<TGamma,TSigma>::GLLiM<TGamma,TSigma>(unsigned D, unsigned L, unsigned K, GLLiMParameters<TGamma,TSigma> &theta, GLLiMConstraints &constraints) {}

template <typename TGamma, typename TSigma>
GLLiM<TGamma, TSigma>::GLLiM(unsigned L, unsigned D, unsigned K, const std::string &gamma_type, const std::string &sigma_type) : theta(L, D, K), theta_star(D, L, K), constraints(gamma_type, sigma_type)
{
    std::cout << "GLLiM Parameters initialized" << std::endl;
    std::cout << GLLiM<TGamma, TSigma>::getDimensions() << std::endl;
    std::cout << GLLiM<TGamma, TSigma>::getConstraints() << std::endl;
}

// ============================== Main public methods ==============================

// void GLLiM<TGamma,TSigma>::initialize(const mat &x, const mat &y, unsigned seed, unsigned nb_iter_EM, unsigned nb_experiences, unsigned max_iteration, double ratio_ll, double floor, unsigned kmeans_iteration, unsigned em_iteration, double floor)
// {
// }

// void GLLiM<TGamma,TSigma>::train(const mat &x, const mat &y, unsigned kmeans_iteration, unsigned em_iteration, double floor)
// void GLLiM<TGamma,TSigma>::train(const mat &x, const mat &y, unsigned max_iteration, double ratio_ll, double floor)
// {
// }

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::train(const mat &x, const mat &y, unsigned max_iteration, double ratio_ll, double floor)
{
    // this->checkConstraints(); // ? Check if Params are valid and update constraints

    if constexpr (std::is_same<TGamma, FullCovariance>::value && std::is_same<TSigma, FullCovariance>::value) // C++17 improvment. "if constexpr" is evaluated at compile time.
    {
        std::cout << "Joint GMM training" << std::endl;
        // GLLiM is equivalent to a classic GMM on the joint law (X,Y). Applying the Armadillo built-in EM method is more efficient.

        JGMM estimator;
        unsigned kmeans_iteration = 10; // TODO set to 0 ? variable in arguments ?
        unsigned em_iteration = max_iteration;
        this->theta = estimator.train(x, y, this->theta, kmeans_iteration, em_iteration, floor); //  comment faire avec les paramètres ?
    }
    else
    {
        std::cout << "GLLiM-EM training" << std::endl;
        EmEstimator<TGamma, TSigma> estimator;
        this->theta = estimator.train(x, y, this->theta, max_iteration, ratio_ll, floor); //  comment faire avec les paramètres ?
    }
}

// returns posterior mean estimates E[yn|xn;θ]
// TODO write formula from Delaforge 2014
template <typename TGamma, typename TSigma>
PredictionResult GLLiM<TGamma, TSigma>::directDensities(const mat &x, const vec &x_incertitude)
{
    unsigned N_obs = x.n_cols;
    PredictionResult result(N_obs, this->theta.D, this->theta.K);

    // ==================== Alter theta covariance and inverse theta ====================

    GLLiMParameters<TGamma, TSigma> theta_altered = this->theta;
    for (unsigned k = 0; k < theta_altered.K; ++k)
    {
        theta_altered.Gamma[k] += diagmat(pow(x_incertitude, 2));
    }

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

// returns prior mean estimates E[xn|yn;θ] when the observation incertitude is different for each observation (no parallelisation)
template <typename TGamma, typename TSigma>
PredictionResult GLLiM<TGamma, TSigma>::inverseDensities(const mat &y, const mat &y_incertitude)
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
        result = GLLiM<TGamma, TSigma>::inverseDensitiesOneInversion(y, vec(y_incertitude));
    }
    else
    {
#pragma omp parallel for
        for (size_t n = 0; n < N_obs; n++)
        {
            PredictionResult res_n = GLLiM<TGamma, TSigma>::inverseDensitiesOneInversion(mat(y.col(n)), vec(y_incertitude.col(n)));
            result.meanPredResult.gmm_weights.row(n) = res_n.meanPredResult.gmm_weights; // (N_obs, K)
            result.meanPredResult.gmm_means.row(n) = res_n.meanPredResult.gmm_means;     // (N_obs, D, K)
            result.meanPredResult.gmm_covs = res_n.meanPredResult.gmm_covs;              // (D, D, K) // TODO Problem because for this case gmm_covs = theta_star.Sigma and is different for each observation :/
            result.meanPredResult.mean.row(n) = res_n.meanPredResult.mean;               // (N_obs, D)
            result.meanPredResult.variance.row(n) = res_n.meanPredResult.variance;       // (N_obs, D, D)
        }
    }

    return result;
}

// void GLLiM<TGamma,TSigma>::getInsights()
// {
// }

// void GLLiM<TGamma,TSigma>::checkConstraints()
// {
// }

// ============================== Getters ==============================

template <typename TGamma, typename TSigma>
std::string GLLiM<TGamma, TSigma>::getDimensions()
{
    std::string str = "GLLiM dimensions are (L=" + std::to_string(this->theta.L) + ", D=" + std::to_string(this->theta.D) + ", K=" + std::to_string(this->theta.K) + ")";
    return str;
}

template <typename TGamma, typename TSigma>
std::string GLLiM<TGamma, TSigma>::getConstraints()
{
    std::string str = "GLLiM constraints are :\n\tgamma_type = '" + this->constraints.gamma_type + "',\n\tsigma_type = '" + this->constraints.sigma_type + "'.";
    return str;
}

template <typename TGamma, typename TSigma>
GLLiMParameters<TGamma, TSigma> GLLiM<TGamma, TSigma>::getParams()
{
    return this->theta;
}

template <typename TGamma, typename TSigma>
GLLiMParametersArma<TGamma, TSigma> GLLiM<TGamma, TSigma>::getParamsArma()
{
    GLLiMParametersArma<TGamma, TSigma> theta_arma(theta.L, theta.D, theta.K);
    theta_arma.Pi = this->theta.Pi;
    theta_arma.A = this->theta.A;
    theta_arma.B = this->theta.B;
    theta_arma.C = this->theta.C;
    theta_arma.Gamma = convertVectorOfCovToArma<TGamma>(this->theta.K, this->theta.L, theta.Gamma);
    theta_arma.Sigma = convertVectorOfCovToArma<TSigma>(this->theta.K, this->theta.D, theta.Sigma);
    return theta_arma;
}

template <typename TGamma, typename TSigma>
rowvec GLLiM<TGamma, TSigma>::getParamPi()
{
    return this->theta.Pi;
}

template <typename TGamma, typename TSigma>
cube GLLiM<TGamma, TSigma>::getParamA()
{
    return this->theta.A;
}

template <typename TGamma, typename TSigma>
mat GLLiM<TGamma, TSigma>::getParamB()
{
    return this->theta.B;
}

template <typename TGamma, typename TSigma>
mat GLLiM<TGamma, TSigma>::getParamC()
{
    return this->theta.C;
}

template <typename TGamma, typename TSigma>
std::vector<TGamma> GLLiM<TGamma, TSigma>::getParamGamma()
{
    return this->theta.Gamma;
}

template <typename TGamma, typename TSigma>
typename TGamma::Type GLLiM<TGamma, TSigma>::getParamGammaArma()
{
    return convertVectorOfCovToArma<TGamma>(theta.K, theta.L, theta.Gamma);
}

template <typename TGamma, typename TSigma>
std::vector<TSigma> GLLiM<TGamma, TSigma>::getParamSigma()
{
    return this->theta.Sigma;
}

template <typename TGamma, typename TSigma>
typename TSigma::Type GLLiM<TGamma, TSigma>::getParamSigmaArma()
{
    return convertVectorOfCovToArma(theta.K, theta.D, theta.Sigma);
}

template <typename TGamma, typename TSigma>
GLLiMParameters<FullCovariance, FullCovariance> GLLiM<TGamma, TSigma>::getInverse()
{
    this->theta_star = this->inverse(this->theta);
    return this->theta_star;
}

template <typename TGamma, typename TSigma>
GLLiMParametersArma<FullCovariance, FullCovariance> GLLiM<TGamma, TSigma>::getInverseArma()
{
    GLLiMParameters<FullCovariance, FullCovariance> theta_star = getInverse();
    GLLiMParametersArma<FullCovariance, FullCovariance> theta_star_arma(theta_star.L, theta_star.D, theta_star.K);
    theta_star_arma.Pi = theta_star.Pi;
    theta_star_arma.A = theta_star.A;
    theta_star_arma.B = theta_star.B;
    theta_star_arma.C = theta_star.C;
    theta_star_arma.Gamma = convertVectorOfCovToArma<FullCovariance>(theta_star.K, theta_star.L, theta_star.Gamma);
    theta_star_arma.Sigma = convertVectorOfCovToArma<FullCovariance>(theta_star.K, theta_star.D, theta_star.Sigma);
    return theta_star_arma;
}
// TODO Is the theta_star attribute really useful if it is recalculated every time and is differring from theta_star_altered

// ============================== Setters ==============================

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParams(const GLLiMParameters<TGamma, TSigma> &theta)
{
    this->theta = theta;
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamsArma(const GLLiMParametersArma<TGamma, TSigma> &theta)
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
    if (!(arma::size(theta.Gamma) == arma::size(TGamma::getTypeSize(this->theta.K, this->theta.L))))
    {
        err += "Gamma dimensions must be of shape (" + std::to_string(this->theta.L) + "," + std::to_string(this->theta.L) + "," + std::to_string(this->theta.K) + ")\n";
    }
    if (!(arma::size(theta.B) == arma::SizeMat(this->theta.D, this->theta.K)))
    {
        err += "B dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.K) + ")\n";
    }
    if (!(arma::size(theta.Sigma) == arma::size(TSigma::getTypeSize(this->theta.K, this->theta.D))))
    {
        err += "Sigma dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.D) + "," + std::to_string(this->theta.K) + ")\n";
    }
    if (err == "")
    {
        this->theta.Pi = theta.Pi;
        this->theta.A = theta.A;
        this->theta.B = theta.B;
        this->theta.C = theta.C;
        this->theta.Gamma = convertArmaToVectorOfCov<TGamma>(this->theta.K, this->theta.L, theta.Gamma);
        this->theta.Sigma = convertArmaToVectorOfCov<TSigma>(this->theta.K, this->theta.D, theta.Sigma);
    }
    else
    {
        throw std::invalid_argument(err);
    }
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamPi(const rowvec &Pi)
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

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamA(const cube &A)
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

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamB(const mat &B)
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

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamC(const mat &C)
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

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamGamma(const std::vector<TGamma> &Gamma)
{

    this->theta.Gamma = Gamma;
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamGammaArma(const typename TGamma::Type &Gamma)
{
    if (arma::size(Gamma) == arma::size(TGamma::getTypeSize(this->theta.K, this->theta.L)))
    {
        this->theta.Gamma = convertArmaToVectorOfCov<TGamma>(this->theta.K, this->theta.L, Gamma);
    }
    else
    {
        throw std::invalid_argument("Gamma dimensions must be of shape (" + std::to_string(this->theta.K) + "," + std::to_string(this->theta.L) + "," + std::to_string(this->theta.L) + ")\n");
    }
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamSigma(const std::vector<TSigma> &Sigma)
{
    this->theta.Sigma = Sigma;
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamSigmaArma(const typename TSigma::Type &Sigma)
{
    if (arma::size(Sigma) == arma::size(TSigma::getTypeSize(this->theta.K, this->theta.D)))
    {
        this->theta.Sigma = convertArmaToVectorOfCov<TSigma>(this->theta.K, this->theta.D, Sigma);
    }
    else
    {
        throw std::invalid_argument("Sigma dimensions must be of shape (" + std::to_string(this->theta.D) + "," + std::to_string(this->theta.D) + "," + std::to_string(this->theta.K) + ")\n");
    }
}

// ============================== Private methods ==============================

template <typename TGamma, typename TSigma>
GLLiMParameters<FullCovariance, FullCovariance> GLLiM<TGamma, TSigma>::inverse(GLLiMParameters<TGamma, TSigma> &theta)
{
    GLLiMParameters<FullCovariance, FullCovariance> theta_star(theta.D, theta.L, theta.K);
    for (unsigned k = 0; k < theta.K; k++)
    {
        if (theta.Pi(k) != 0)
        {
            theta_star.Pi(k) = theta.Pi(k);
            TSigma sigma_inv = theta.Sigma[k].inv();
            TGamma gamma_inv = theta.Gamma[k].inv();
            theta_star.C.col(k) = theta.A.slice(k) * theta.C.col(k) + theta.B.col(k);
            theta_star.Gamma[k] = FullCovariance(theta.Sigma[k] + theta.A.slice(k) * theta.Gamma[k] * theta.A.slice(k).t());
            theta_star.Sigma[k] = FullCovariance((gamma_inv + mat(theta.A.slice(k).t()) * sigma_inv * mat(theta.A.slice(k))).i());
            theta_star.A.slice(k) = theta_star.Sigma[k] * mat(theta.A.slice(k).t()) * sigma_inv;
            theta_star.B.col(k) = theta_star.Sigma[k] * vec(gamma_inv * vec(theta.C.col(k)) - mat(theta.A.slice(k).t()) * sigma_inv * vec(theta.B.col(k)));
        }
    }
    return theta_star;
}

template <typename TGamma, typename TSigma>
template <typename TGamma2, typename TSigma2>
std::tuple<mat, cube, cube> GLLiM<TGamma, TSigma>::constructGMM(const mat &x, GLLiMParameters<TGamma2, TSigma2> &theta)
{
    unsigned N_obs = x.n_cols;

    // Compute weights
    mat weights(N_obs, theta.K);
    double log_det_gamma;
    vec x_u;

#pragma omp parallel for
    for (unsigned k = 0; k < theta.K; k++)
    {
        if (theta.Pi(k) == 0)
        {
            weights.col(k) = -datum::inf;
        }
        else
        {
            log_det_gamma = theta.Gamma[k].log_det();
            for (unsigned n = 0; n < N_obs; n++)
            {
                x_u = x.col(n) - theta.C.col(k);
                if (log_det_gamma != -datum::inf)
                {
                    weights(n, k) = log(theta.Pi(k)) - 0.5 * (theta.L * log(2 * datum::pi) + log_det_gamma + dot((rowvec(x_u.t()) * theta.Gamma[k].inv()).t(), x_u));
                }
            }
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
    cube covariances(theta.D, theta.D, theta.K); // The covariance is indenpendent from x
    for (unsigned k = 0; k < theta.K; k++)
    {
        covariances.slice(k) = theta.Sigma[k].get_mat();
    }

    return std::make_tuple(weights, means, covariances);
}

// returns prior mean estimates E[xn|yn;θ]
template <typename TGamma, typename TSigma>
PredictionResult GLLiM<TGamma, TSigma>::inverseDensitiesOneInversion(const mat &y, const vec &y_incertitude)
// TODO merge this method with directDensities. Check out the differences
{
    unsigned N_obs = y.n_cols;
    PredictionResult result(N_obs, this->theta.L, this->theta.K);

    // ==================== Alter theta covariance and inverse theta ====================

    GLLiMParameters<TGamma, TSigma> theta_altered = this->theta;
    for (unsigned k = 0; k < theta_altered.K; ++k)
    {
        theta_altered.Sigma[k] += diagmat(pow(y_incertitude, 2));
    }
    GLLiMParameters<FullCovariance, FullCovariance> theta_star_altered = inverse(theta_altered);

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

// ============================== Explicit instantiation of template classes ==============================

template class GLLiM<FullCovariance, FullCovariance>;
template class GLLiM<FullCovariance, DiagCovariance>;
template class GLLiM<FullCovariance, IsoCovariance>;
template class GLLiM<DiagCovariance, FullCovariance>;
template class GLLiM<DiagCovariance, DiagCovariance>;
template class GLLiM<DiagCovariance, IsoCovariance>;
template class GLLiM<IsoCovariance, FullCovariance>;
template class GLLiM<IsoCovariance, DiagCovariance>;
template class GLLiM<IsoCovariance, IsoCovariance>;
