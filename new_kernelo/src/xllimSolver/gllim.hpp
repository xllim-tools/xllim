#ifndef GLLIM_HPP
#define GLLIM_HPP

#include <armadillo>

using namespace arma;

struct GLLiMParameters
{
    // TODO : proper definitions and documentation
    unsigned L; // The dimension of the model input (number of features) that should corresponds to the low dimension value
    unsigned D; // The dimension of the model output that should corresponds to the high dimension value
    unsigned K; // The number of affine transformation which corresponds also to the number of gaussian distributions in the mixture
    rowvec Pi;  // A row vector of size K containing the weights of the gaussian distributions in the mixture
    cube A;     // A cube of size (D,L,K)
    mat C;      // A matrix of size (L,K) containing the means of the mixture of gaussian distribution that define low dimension data
    cube Gamma; // A cube of size (L,L,,K) containing the covariance matrices of the mixture of gaussian distribution that define low dimension data
    mat B;      // A matrix of size (D,K)
    cube Sigma; // A cube of size (D,D,K) containing the covariance matrices of the mixture of gaussian distribution that define high dimension data

    // For more information on these parameters, see the formula 2 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
    // High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.

    GLLiMParameters(unsigned L, unsigned D, unsigned K) : L(L), D(D), K(K), Pi(K), A(D, L, K), C(L, K), Gamma(L, L, K), B(D, K), Sigma(D, D, K) {}
};

struct MeanPredictionResult
{
    mat mean;        // The mean of the GMM which stands for the prediction (N_obs, D)
    cube variance;   // The variance of the prediction (N_obs, D, D)
    mat gmm_weights; // The weights of the components of the GMM (N_obs, K)
    cube gmm_means;  // The means of each component in the GMM (N_obs, D, K)
    cube gmm_covs;   // The covariance matrices of each component in the GMM (D, D, K)

    MeanPredictionResult(unsigned N_obs, unsigned D, unsigned K) : mean(N_obs, D), variance(N_obs, D, D), gmm_weights(N_obs, K), gmm_means(N_obs, D, K), gmm_covs(D, D, K) {}
};

struct CenterPredictionResult
{
    vec weights; // The weights of the centers
    mat means;   // The centers that stands for the predictions
    cube covs;   // The covariance matrices of the centers
};

struct PredictionResult
{
    MeanPredictionResult meanPredResult;     // @see MeanPredictionResult MeanPredictionResult
    CenterPredictionResult centerPredResult; // @see CenterPredictionResult CenterPredictionResult

    PredictionResult(unsigned N_obs, unsigned D, unsigned K) : meanPredResult(N_obs, D, K) {}
};

class GLLiM
{
public:
    /**
     * TODO
     */
    // GLLiM(unsigned D, unsigned L, unsigned K, GLLiMParameters &theta, GLLiMConstraints &constraints);
    GLLiM(unsigned L, unsigned D, unsigned K); // création de la classe (de theta)

    // void initialize(
    //     const mat &x,
    //     const mat &y,
    //     unsigned seed,
    //     unsigned nb_iter_EM = 1,     // default = FixedInit
    //     unsigned nb_experiences = 1, // default = FixedInit
    //     // EMLearningConfig (for diagonal cov)
    //     unsigned max_iteration,
    //     double ratio_ll,
    //     double floor,
    //     // GMMLearningConfig (for full cov)
    //     unsigned kmeans_iteration,
    //     unsigned em_iteration,
    //     double floor);
    // void train(const mat &x, const mat &y, unsigned max_iteration, double ratio_ll, double floor);
    void train(const mat &x, const mat &y, unsigned kmeans_iteration, unsigned em_iteration, double floor);

    GLLiMParameters getParams();
    std::string getDimensions();
    rowvec getParamPi();
    cube getParamA();
    mat getParamC();
    cube getParamGamma();
    mat getParamB();
    cube getParamSigma();

    void setParams(const GLLiMParameters &theta);
    void setParamPi(const rowvec &Pi);
    void setParamA(const cube &A);
    void setParamC(const mat &C);
    void setParamGamma(const cube &Gamma);
    void setParamB(const mat &B);
    void setParamSigma(const cube &Sigma);

    GLLiMParameters getInverse();

    PredictionResult directDensities(const mat &x, const vec &x_incertitude);
    PredictionResult directDensities(const mat &x)
    {
        return directDensities(x, vec(theta.L, fill::zeros));
    };
    PredictionResult inverseDensities(const mat &y, const mat &y_incertitude);
    PredictionResult inverseDensitiesOneInversion(const mat &y, const vec &y_incertitude);
    PredictionResult inverseDensities(const mat &y)
    {
        return inverseDensitiesOneInversion(y, vec(theta.L, fill::zeros));
    };

    // void getInsights();

private:
    // TODO
    GLLiMParameters theta;      // The parameters of the direct GLLiM model
    GLLiMParameters theta_star; // The parameters of the inverse GLLiM model
    GLLiMParameters inverse(GLLiMParameters &theta);
    std::tuple<mat, cube, cube> constructGMM(const mat &x, GLLiMParameters &theta);
    // std::shared_ptr<Iinitilizer<T, U>> initializer;                                            /**< @see Iinitilizer Iinitilizer*/
    // std::shared_ptr<Iestimator<T, U>> estimator;                                               /**< @see Iestimator Iestimator*/
    // std::shared_ptr<GLLiMParameters<T, U>> gllim_parameters;                                   /**< The parameters of the direct GLLiM model*/
    // std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> inverse_gllim_parameters; /**< The parameters of the inverse GLLiM model*/
    // unsigned K;                                                                                /**< the number of affine transformation and the number of gaussian distributions in the mixture */

protected:
    // TODO
    // /**
    //  * This method adjusts the Sigma parameter of the trained GLLiM model with the variance of the observation before computing the corresponding GMM
    //  * @param gllim : the parameters of the trained GLLiM
    //  * @param cov_obs : the variance or measure error of the observation
    //  */
    // void alterCovariance(GLLiMParameters<T, U> &gllim, const vec &cov_obs);
};

#endif // GLLIM_HPP
