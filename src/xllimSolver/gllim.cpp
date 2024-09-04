#include "gllim.hpp"
#include "jgmm.hpp"
#include "emEstimator.hpp"
#include "../generator/RandomGenerator.hpp"
#include "../utils/utils.hpp"
#include "../logging/logger.hpp"

// ============================== Static methods ==============================

template <typename TCov>
static typename TCov::Type convertVectorOfCovToArray(unsigned K, unsigned dimension, std::vector<TCov> cov_vector)
{
    typename TCov::Type cov_array(TCov::getTypeSize(K, dimension));
    for (unsigned k = 0; k < K; k++)
    {
        cov_array.row(k) = cov_vector[k].get();
    }
    return cov_array;
}

template <typename TCov>
static std::vector<TCov> convertArrayToVectorOfCov(unsigned &K, unsigned &dimension, const typename TCov::Type &cov_array)
{
    std::vector<TCov> cov_vector(K, TCov(dimension));
    for (unsigned k = 0; k < K; k++)
    {
        cov_vector[k] = TCov(cov_array.row(k));
    }
    return cov_vector;
}

// ============================== Constructors ==============================

// GLLiM<TGamma,TSigma>::GLLiM<TGamma,TSigma>(unsigned D, unsigned L, unsigned K, GLLiMParameters<TGamma,TSigma> &theta, GLLiMConstraints &constraints) {}

template <typename TGamma, typename TSigma>
GLLiM<TGamma, TSigma>::GLLiM(unsigned K, unsigned D, unsigned L, const std::string &gamma_type, const std::string &sigma_type, unsigned n_hidden_variables) : theta_(K, D, L - n_hidden_variables, n_hidden_variables), constraints_(gamma_type, sigma_type)
{
    Logger::getInstance().log(INFO, "GLLiM Parameters initialized");
    Logger::getInstance().log(INFO, GLLiM<TGamma, TSigma>::getDimensions());
    Logger::getInstance().log(INFO, GLLiM<TGamma, TSigma>::getConstraints());
}

// ============================== Main public methods ==============================

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::initialize(const mat &t, const mat &y, unsigned gllim_em_iteration, double gllim_em_floor, unsigned gmm_kmeans_iteration, unsigned gmm_em_iteration, double gmm_floor, unsigned nb_experiences, unsigned seed, int verbose)
{
    auto start = std::chrono::high_resolution_clock::now();

    unsigned L_t = t.n_rows,
             L_w = theta_.L_w,
             D = y.n_rows,
             K = theta_.K,
             N = t.n_cols;

    // Dimension checking
    if (L_t != theta_.L_t)
    {
        throw std::invalid_argument("The input matrix t must be of dimension L_t = L - n_hidden_variables = " + std::to_string(theta_.L_t));
    }
    if (D != theta_.D)
    {
        throw std::invalid_argument("The output matrix y must be of dimension D = " + std::to_string(theta_.D));
    }
    if (N != y.n_cols)
    {
        throw std::invalid_argument("The input and output matrices must have same number of observations N.");
    }

    double best_log_likelihood = -(datum::inf);
    double log_likelihood;
    GLLiMParameters<TGamma, TSigma> best_theta(K, D, L_t, L_w);
    GLLiMParameters<TGamma, TSigma> local_theta(K, D, L_t, L_w);
    mat log_r(N, K, fill::zeros);

    rowvec gmm_weights(K);
    mat gmm_means(L_t, K);
    cube gmm_covs(L_t, L_t, K);
    JGMM gmmEstimator;
    EmEstimator<TGamma, TSigma> gllimEmEstimator;
    DataGeneration::RandomGenerator randomGenerator(seed);

    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "Start Initialization");
    }
    for (unsigned exp = 0; exp < nb_experiences; exp++)
    {
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "Initialisation : " + std::to_string(exp + 1));
        }

        // generate a mean for the GMM using a data generator strategy
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "\tGenerate GMM means");
        }
        randomGenerator.execute(gmm_means);

        // use the same weight for all the clusters
        gmm_weights.ones();
        gmm_weights /= K;

        // Create a cube of K covariance matrices with a homothety constraint
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "\tGenerate GMM covariance matrices");
        }
        gmm_covs.zeros();
        for (unsigned k = 0; k < K; k++)
        {
            gmm_covs.slice(k).diag() += sqrt(1.0 / (pow(K, 1.0 / L_t)));
        }

        // train a GMM over nb_iter iteration
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "\tTrain the GMM model");
        }
        gmm_full gmm;
        gmm.set_params(gmm_means, gmm_covs, gmm_weights);
        gmm.learn(t, K, maha_dist, keep_existing, gmm_kmeans_iteration, gmm_em_iteration, gmm_floor, false);

        // compute log_rnk using the posterior of the GMM after the training
        for (unsigned k = 0; k < K; k++)
        {
            log_r.col(k) = gmm.log_p(t, k).t();
        }

        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "\tCompute Initial theta vector of the GLLiM model from GMM");
        }
        // Hybrid model : latent variables follow a normal distribution
        cube mu_w_normal_distr(L_w, N, K, fill::zeros);
        cube S_w_normal_distr(L_w, L_w, K, fill::zeros);
        S_w_normal_distr.each_slice() = mat(L_w, L_w, fill::eye);
        gllimEmEstimator.maximization_step(t, y, local_theta, log_r, mu_w_normal_distr, S_w_normal_distr, gllim_em_floor);

        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "\tTrain the initial GLLiM model");
        }
        // ! Simplification : just use train() method. If ratio_ll is set to 0, the new version it is equivalent (must be verified) to the old version.
        gllimEmEstimator.train(t, y, local_theta, gllim_em_iteration, -1.0, gllim_em_floor, verbose); // TODO verbose = 0,1,2..

        vec log_likelihood_list = gllimEmEstimator.get_log_likelihood();      // log_likelihood for each iteration
        log_likelihood = log_likelihood_list[log_likelihood_list.n_elem - 1]; // log_likelihood of last iteration

        if (log_likelihood >= best_log_likelihood)
        {
            best_theta = local_theta;
            best_log_likelihood = log_likelihood;
        }
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "\tCurrent log likelihood : " + std::to_string(log_likelihood) + ", Best log likelihood : " + std::to_string(best_log_likelihood));
        }
    }

    theta_ = best_theta;

    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "FinishInitialization");
    }

    // Save relevant information about initialisation within Insights structure
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    insights_.initialisation = InitialisationInsights(duration, start, end, N, gllim_em_iteration, gllim_em_floor, gmm_kmeans_iteration, gmm_em_iteration, gmm_floor, nb_experiences);
}

// void GLLiM<TGamma,TSigma>::train(const mat &x, const mat &y, unsigned kmeans_iteration, unsigned em_iteration, double floor)
// void GLLiM<TGamma,TSigma>::train(const mat &x, const mat &y, unsigned max_iteration, double ratio_ll, double floor)
// {
// }

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::train(const mat &x, const mat &y, unsigned max_iteration, double ratio_ll, double floor, int verbose)
{
    // checkConstraints(); // ? Check if Params are valid and update constraints

    auto start = std::chrono::high_resolution_clock::now();

    if constexpr (std::is_same<TGamma, FullCovariance>::value && std::is_same<TSigma, FullCovariance>::value) // C++17 improvment. "if constexpr" is evaluated at compile time.
    {
        if (verbose >= 0)
        {
            Logger::getInstance().log(WARNING, "A classic GMM training is applied on the equivalent joint-GMM to GLLiM. The algorithm is provided by the Armadillo library. This option is only available whith 'full/full' constraints. The training is equivalent and faster than the GLLiM-EM algorithm.");
        }
        // GLLiM is equivalent to a classic GMM on the joint law (X,Y). Applying the Armadillo built-in EM method is more efficient.

        JGMM estimator;
        unsigned kmeans_iteration = 10; // TODO set to 0 ? variable in arguments ?
        unsigned em_iteration = max_iteration;
        estimator.train(x, y, theta_, kmeans_iteration, em_iteration, floor, verbose); //  comment faire avec les paramètres ?
    }
    else
    {
        EmEstimator<TGamma, TSigma> estimator;
        // TODO estimator.train returnind void is better
        estimator.train(x, y, theta_, max_iteration, ratio_ll, floor, verbose); //  comment faire avec les paramètres ?
        insights_.log_likelihood = estimator.get_log_likelihood();              // Save relevant information about training within Insights structure
    }

    // Save relevant information about training within Insights structure
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    insights_.training = TrainingInsights(duration, start, end, x.n_cols, max_iteration, ratio_ll, floor);
}

// returns posterior mean estimates E[yn|xn;θ]
// TODO write formula from Delaforge 2014
template <typename TGamma, typename TSigma>
PredictionResult GLLiM<TGamma, TSigma>::directDensities(const mat &x, const vec &x_incertitude, int verbose)
{
    unsigned N_obs = x.n_cols;
    PredictionResult result(N_obs, theta_.D, theta_.K);

    // ==================== Alter theta covariance and inverse theta ====================

    GLLiMParameters<TGamma, TSigma> theta_altered = theta_;

    if (!x_incertitude.is_zero())
    {
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "Alter theta covariance");
        }
        for (unsigned k = 0; k < theta_.K; ++k)
        {
            theta_altered.Gamma[k] += diagmat(pow(x_incertitude, 2));
        }
    }

    // ==================== Construct the GMM of the forward conditional model ====================

    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "Construct the GMM of the forward conditional model");
    }
    std::tuple<mat, cube, cube> GMMs = constructGMM(x, theta_altered);
    result.meanPredResult.gmm_weights = std::get<0>(GMMs); // (N_obs, K)
    result.meanPredResult.gmm_means = std::get<1>(GMMs);   // (N_obs, D, K)
    result.meanPredResult.gmm_covs = std::get<2>(GMMs);    // (D, D, K)

    // ============================= Prediction mean estimations ==================================

    // Compute the mean of the means in the mixture
    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "Compute the weighted mean of the means in the mixture");
    }
    result.meanPredResult.mean = mat(N_obs, theta_.D);
    for (unsigned k = 0; k < theta_.K; ++k)
    {
        result.meanPredResult.mean += diagmat(result.meanPredResult.gmm_weights.col(k)) * result.meanPredResult.gmm_means.slice(k);
    }

    // Compute the mean of covariances in the mixture
    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "Compute the weighted covariance of the covariances in the mixture");
    }
// TODO voir formule dans Kugler 2021. Can be simplified
#pragma omp parallel
    result.meanPredResult.variance = cube(N_obs, theta_.D, theta_.D);
    for (unsigned n = 0; n < N_obs; ++n)
    {
        for (unsigned k = 0; k < theta_.K; ++k)
        {
            rowvec mean_diff = result.meanPredResult.gmm_means.slice(k).row(n) - result.meanPredResult.mean.row(n);
            result.meanPredResult.variance.row(n) += result.meanPredResult.gmm_weights(n, k) * (result.meanPredResult.gmm_covs.slice(k) + mean_diff.t() * mean_diff);
        }
    }

    return result;
}

// returns prior mean estimates E[xn|yn;θ] when the observation incertitude is different for each observation (no parallelisation)
template <typename TGamma, typename TSigma>
PredictionResult GLLiM<TGamma, TSigma>::inverseDensities(const mat &y, const mat &y_incertitude, int verbose)
// TODO merge this method with directDensities. Check out the differences
{
    unsigned N_obs = y.n_cols;
    PredictionResult result(N_obs, theta_.L, theta_.K);

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
        result = GLLiM<TGamma, TSigma>::inverseDensitiesOneInversion(y, vec(y_incertitude), verbose);
    }
    else
    {
#pragma omp parallel for
        for (size_t n = 0; n < N_obs; n++)
        {
            PredictionResult res_n = GLLiM<TGamma, TSigma>::inverseDensitiesOneInversion(mat(y.col(n)), vec(y_incertitude.col(n)), verbose);
            result.meanPredResult.gmm_weights.row(n) = res_n.meanPredResult.gmm_weights; // (N_obs, K)
            result.meanPredResult.gmm_means.row(n) = res_n.meanPredResult.gmm_means;     // (N_obs, D, K)
            result.meanPredResult.gmm_covs = res_n.meanPredResult.gmm_covs;              // (D, D, K) // TODO Problem because for this case gmm_covs = theta_star.Sigma and is different for each observation :/
            result.meanPredResult.mean.row(n) = res_n.meanPredResult.mean;               // (N_obs, D)
            result.meanPredResult.variance.row(n) = res_n.meanPredResult.variance;       // (N_obs, D, D)
        }
    }

    return result;
}

template <typename TGamma, typename TSigma>
Insights GLLiM<TGamma, TSigma>::getInsights()
{
    insights_.time = std::chrono::seconds(0);
    if (insights_.initialisation.N_obs != 0)
    {
        insights_.time += insights_.initialisation.time;
    }
    else
    {
        Logger::getInstance().log(WARNING, "GLLiM model has not been initialized");
    }
    if (insights_.training.N_obs != 0)
    {
        insights_.time += insights_.training.time;
    }
    else
    {
        Logger::getInstance().log(WARNING, "GLLiM model has not been trained");
    }
    return insights_;
}

// void GLLiM<TGamma,TSigma>::checkConstraints()
// {
// }

// ============================== Getters ==============================

template <typename TGamma, typename TSigma>
std::string GLLiM<TGamma, TSigma>::getDimensions()
{
    std::string str = "GLLiM dimensions are (L=" + std::to_string(theta_.L) + ", D=" + std::to_string(theta_.D) + ", K=" + std::to_string(theta_.K) + ")";
    return str;
}

template <typename TGamma, typename TSigma>
std::string GLLiM<TGamma, TSigma>::getConstraints()
{
    std::string str = "GLLiM constraints are :\n\tgamma_type = '" + constraints_.gamma_type + "',\n\tsigma_type = '" + constraints_.sigma_type + "'.";
    return str;
}

template <typename TGamma, typename TSigma>
GLLiMParameters<TGamma, TSigma> GLLiM<TGamma, TSigma>::getParams()
{
    return theta_;
}

template <typename TGamma, typename TSigma>
GLLiMParametersArray<TGamma, TSigma> GLLiM<TGamma, TSigma>::getParamsArray()
{
    GLLiMParametersArray<TGamma, TSigma> theta_arma(theta_.K, theta_.D, theta_.L);
    theta_arma.Pi = theta_.Pi;
    theta_arma.B = theta_.B.t();
    theta_arma.C = theta_.C.t();
    theta_arma.Gamma = convertVectorOfCovToArray<TGamma>(theta_.K, theta_.L, theta_.Gamma);
    theta_arma.Sigma = convertVectorOfCovToArray<TSigma>(theta_.K, theta_.D, theta_.Sigma);
    for (size_t i = 0; i < theta_.A.n_slices; ++i)
    {
        theta_arma.A.row(i) = theta_.A.slice(i);
    }
    return theta_arma;
}

template <typename TGamma, typename TSigma>
rowvec GLLiM<TGamma, TSigma>::getParamPi()
{
    return theta_.Pi;
}

template <typename TGamma, typename TSigma>
cube GLLiM<TGamma, TSigma>::getParamA()
{
    cube A(theta_.A.n_slices, theta_.A.n_rows, theta_.A.n_cols);
    for (size_t i = 0; i < A.n_rows; ++i)
    {
        A.row(i) = theta_.A.slice(i);
    }
    return A;
}

template <typename TGamma, typename TSigma>
mat GLLiM<TGamma, TSigma>::getParamB()
{
    return theta_.B.t();
}

template <typename TGamma, typename TSigma>
mat GLLiM<TGamma, TSigma>::getParamC()
{
    return theta_.C.t();
}

template <typename TGamma, typename TSigma>
std::vector<TGamma> GLLiM<TGamma, TSigma>::getParamGamma()
{
    return theta_.Gamma;
}

template <typename TGamma, typename TSigma>
typename TGamma::Type GLLiM<TGamma, TSigma>::getParamGammaArray()
{
    return convertVectorOfCovToArray<TGamma>(theta_.K, theta_.L, theta_.Gamma);
}

template <typename TGamma, typename TSigma>
std::vector<TSigma> GLLiM<TGamma, TSigma>::getParamSigma()
{
    return theta_.Sigma;
}

template <typename TGamma, typename TSigma>
typename TSigma::Type GLLiM<TGamma, TSigma>::getParamSigmaArray()
{
    return convertVectorOfCovToArray(theta_.K, theta_.D, theta_.Sigma);
}

template <typename TGamma, typename TSigma>
GLLiMParameters<FullCovariance, FullCovariance> GLLiM<TGamma, TSigma>::getInverse()
{
    return inverse(theta_);
}

template <typename TGamma, typename TSigma>
GLLiMParametersArray<FullCovariance, FullCovariance> GLLiM<TGamma, TSigma>::getInverseArray()
{
    GLLiMParameters<FullCovariance, FullCovariance> theta_star = getInverse();
    GLLiMParametersArray<FullCovariance, FullCovariance> theta_star_array(theta_star.K, theta_star.D, theta_star.L);
    theta_star_array.Pi = theta_star.Pi;
    theta_star_array.B = theta_star.B.t();
    theta_star_array.C = theta_star.C.t();
    theta_star_array.Gamma = convertVectorOfCovToArray<FullCovariance>(theta_star.K, theta_star.L, theta_star.Gamma);
    theta_star_array.Sigma = convertVectorOfCovToArray<FullCovariance>(theta_star.K, theta_star.D, theta_star.Sigma);
    for (size_t i = 0; i < theta_star.A.n_slices; ++i)
    {
        theta_star_array.A.row(i) = theta_star.A.slice(i);
    }
    return theta_star_array;
}
// TODO Is the theta_star attribute really useful if it is recalculated every time and is differring from theta_star_altered

// ============================== Setters ==============================

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParams(const GLLiMParameters<TGamma, TSigma> &theta)
{
    theta_ = theta;
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamsArray(const GLLiMParametersArray<TGamma, TSigma> &theta)
{
    std::string err = "";
    if (!(theta.Pi.is_vec() && (theta.Pi.n_cols == theta_.K)))
    {
        err += "Pi dimensions must be of shape (" + std::to_string(theta_.K) + ")\n";
    }
    if (!(abs(accu(theta.Pi) - 1.0) < 1e-9))
    {
        err += "The sum of weights must be equal to 1\n";
    }
    if (!(arma::size(theta.A) == arma::SizeCube(theta_.K, theta_.D, theta_.L)))
    {
        err += "A dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.D) + "," + std::to_string(theta_.L) + ")\n";
    }
    if (!(arma::size(theta.C) == arma::SizeMat(theta_.K, theta_.L)))
    {
        err += "C dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.L) + ")\n";
    }
    if (!(arma::size(theta.Gamma) == arma::size(TGamma::getTypeSize(theta_.K, theta_.L))))
    {
        err += "Gamma dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.L) + "," + std::to_string(theta_.L) + ")\n";
    }
    if (!(arma::size(theta.B) == arma::SizeMat(theta_.K, theta_.D)))
    {
        err += "B dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.D) + ")\n";
    }
    if (!(arma::size(theta.Sigma) == arma::size(TSigma::getTypeSize(theta_.K, theta_.D))))
    {
        err += "Sigma dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.D) + "," + std::to_string(theta_.D) + ")\n";
    }
    if (err == "")
    {
        theta_.Pi = theta.Pi;
        theta_.B = theta.B.t();
        theta_.C = theta.C.t();
        theta_.Gamma = convertArrayToVectorOfCov<TGamma>(theta_.K, theta_.L, theta.Gamma);
        theta_.Sigma = convertArrayToVectorOfCov<TSigma>(theta_.K, theta_.D, theta.Sigma);
        for (size_t i = 0; i < theta.A.n_rows; ++i)
        {
            theta_.A.slice(i) = theta.A.row(i);
        }
    }
    else
    {
        throw std::invalid_argument(err);
    }
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamPi(const rowvec &Pi)
{
    if (Pi.is_vec() && (Pi.n_cols == theta_.K))
    {
        if ((abs(accu(Pi) - 1.0) < 1e-9))
        {
            theta_.Pi = Pi;
        }
        else
        {
            throw std::invalid_argument("The sum of weights must be equal to 1.");
        }
    }
    else
    {
        throw std::invalid_argument("Pi dimensions must be of shape (" + std::to_string(theta_.K) + ")");
    }
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamA(const cube &A)
{
    if (arma::size(A) == arma::SizeCube(theta_.K, theta_.D, theta_.L))
    {
        for (size_t i = 0; i < A.n_rows; ++i)
        {
            theta_.A.slice(i) = A.row(i);
        }
    }
    else
    {
        throw std::invalid_argument("A dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.D) + "," + std::to_string(theta_.L) + ")");
    }
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamB(const mat &B)
{
    if (arma::size(B) == arma::SizeMat(theta_.K, theta_.D))
    {
        theta_.B = B.t();
    }
    else
    {
        throw std::invalid_argument("B dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.D) + ")");
    }
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamC(const mat &C)
{
    if (arma::size(C) == arma::SizeMat(theta_.K, theta_.L))
    {
        theta_.C = C.t();
    }
    else
    {
        throw std::invalid_argument("C dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.D) + ")");
    }
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamGamma(const std::vector<TGamma> &Gamma)
{

    theta_.Gamma = Gamma;
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamGammaArray(const typename TGamma::Type &Gamma)
{
    if (arma::size(Gamma) == arma::size(TGamma::getTypeSize(theta_.K, theta_.L)))
    {
        theta_.Gamma = convertArrayToVectorOfCov<TGamma>(theta_.K, theta_.L, Gamma);
    }
    else
    {
        throw std::invalid_argument("Gamma dimensions must be of shape (" + std::to_string(theta_.K) + "," + std::to_string(theta_.L) + "," + std::to_string(theta_.L) + ")\n");
    }
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamSigma(const std::vector<TSigma> &Sigma)
{
    theta_.Sigma = Sigma;
}

template <typename TGamma, typename TSigma>
void GLLiM<TGamma, TSigma>::setParamSigmaArray(const typename TSigma::Type &Sigma)
{
    if (arma::size(Sigma) == arma::size(TSigma::getTypeSize(theta_.K, theta_.D)))
    {
        theta_.Sigma = convertArrayToVectorOfCov<TSigma>(theta_.K, theta_.D, Sigma);
    }
    else
    {
        throw std::invalid_argument("Sigma dimensions must be of shape (" + std::to_string(theta_.D) + "," + std::to_string(theta_.D) + "," + std::to_string(theta_.K) + ")\n");
    }
}

// ============================== Private methods ==============================

template <typename TGamma, typename TSigma>
GLLiMParameters<FullCovariance, FullCovariance> GLLiM<TGamma, TSigma>::inverse(GLLiMParameters<TGamma, TSigma> &theta)
{
    GLLiMParameters<FullCovariance, FullCovariance> theta_star(theta.K, theta.L, theta.D, theta.L_w);
    for (unsigned k = 0; k < theta_.K; k++)
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
        means.slice(k) = (theta.A.slice(k) * x).t();
        means.slice(k).each_row() += theta.B.col(k).t();
        // means.slice(k) = (theta.A.slice(k) * x).t() + arma::repmat(theta.B.col(k).t(), N_obs, 1);
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
PredictionResult GLLiM<TGamma, TSigma>::inverseDensitiesOneInversion(const mat &y, const vec &y_incertitude, int verbose)
// TODO merge this method with directDensities. Check out the differences
{
    unsigned N_obs = y.n_cols;
    PredictionResult result(N_obs, theta_.L, theta_.K);

    // ==================== Alter theta covariance and inverse theta ====================

    GLLiMParameters<TGamma, TSigma> theta_altered = theta_;

    if (!y_incertitude.is_zero())
    {
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "Alter theta covariance");
        }
        for (unsigned k = 0; k < theta_altered.K; ++k)
        {
            theta_altered.Sigma[k] += diagmat(pow(y_incertitude, 2));
        }
    }

    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "Inverse theta");
    }
    GLLiMParameters<FullCovariance, FullCovariance> theta_star_altered = inverse(theta_altered);

    // ==================== Construct the GMM of the inverse conditional model ====================

    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "Construct the GMM of the inverse conditional model");
    }
    std::tuple<mat, cube, cube> GMMs = constructGMM(y, theta_star_altered);
    result.meanPredResult.gmm_weights = std::get<0>(GMMs); // (N_obs, K)
    result.meanPredResult.gmm_means = std::get<1>(GMMs);   // (N_obs, D, K)
    result.meanPredResult.gmm_covs = std::get<2>(GMMs);    // (D, D, K)

    // ============================= Prediction mean estimations ==================================

    // Compute the mean of the means in the mixture
    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "Compute the weighted mean of the means in the mixture");
    }
    result.meanPredResult.mean = mat(N_obs, theta_star_altered.D); // theta_star_altered.D or theta_altered.L (the second one is more explicit. )
    for (unsigned k = 0; k < theta_star_altered.K; ++k)
    {
        result.meanPredResult.mean += diagmat(result.meanPredResult.gmm_weights.col(k)) * result.meanPredResult.gmm_means.slice(k);
    }

    // Compute the mean of covariances in the mixture
    if (verbose >= 1)
    {
        Logger::getInstance().log(INFO, "Compute the weighted covariance of the covariances in the mixture");
    }
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
