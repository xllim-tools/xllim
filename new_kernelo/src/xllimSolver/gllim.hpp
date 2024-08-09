#ifndef GLLIM_HPP
#define GLLIM_HPP

#include "covariances/covariance.hpp"
#include "gllimStructures/gllimParameters.hpp"
#include "gllimStructures/gllimParametersArma.hpp"
#include "gllimStructures/gllimConstraints.hpp"
#include "gllimStructures/predictionResults.hpp"
#include "gllimStructures/insights.hpp"

using namespace arma;

class GLLiMBase
{
public:
    virtual ~GLLiMBase() = default;
};

template <typename TGamma, typename TSigma>
class GLLiM : public GLLiMBase
{
public:
    /**
     * TODO
     */
    GLLiM(unsigned L, unsigned D, unsigned K, const std::string &gamma_type, const std::string &sigma_type); // création de la classe (de theta)
    // GLLiM(unsigned L, unsigned D, unsigned K); // création de la classe (de theta)

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
    void initialize(const mat &t, const mat &y, unsigned gllim_em_iteration, double gllim_em_floor, unsigned gmm_kmeans_iteration, unsigned gmm_em_iteration, double gmm_floor, unsigned nb_experiences, unsigned seed, int verbose = 1);
    void train(const mat &x, const mat &y, unsigned max_iteration, double ratio_ll, double floor, int verbose = 1);
    // void train(const mat &x, const mat &y, unsigned kmeans_iteration, unsigned em_iteration, double floor);

    std::string getDimensions();
    std::string getConstraints();
    GLLiMParameters<TGamma, TSigma> getParams();
    GLLiMParametersArma<TGamma, TSigma> getParamsArma();
    rowvec getParamPi();
    cube getParamA();
    mat getParamB();
    mat getParamC();
    std::vector<TGamma> getParamGamma();
    typename TGamma::Type getParamGammaArma();
    std::vector<TSigma> getParamSigma();
    typename TSigma::Type getParamSigmaArma();

    void setParams(const GLLiMParameters<TGamma, TSigma> &theta);
    void setParamsArma(const GLLiMParametersArma<TGamma, TSigma> &theta);
    void setParamPi(const rowvec &Pi);
    void setParamA(const cube &A);
    void setParamB(const mat &B);
    void setParamC(const mat &C);
    void setParamGamma(const std::vector<TGamma> &Gamma);
    void setParamGammaArma(const typename TGamma::Type &Gamma);
    void setParamSigma(const std::vector<TSigma> &Sigma);
    void setParamSigmaArma(const typename TSigma::Type &Sigma);

    GLLiMParameters<FullCovariance, FullCovariance> getInverse();
    GLLiMParametersArma<FullCovariance, FullCovariance> getInverseArma();

    PredictionResult directDensities(const mat &x, const vec &x_incertitude, int verbose = 1);
    PredictionResult directDensities(const mat &x, int verbose = 1) { return directDensities(x, vec(theta.L, fill::zeros), verbose); };

    PredictionResult inverseDensities(const mat &y, const mat &y_incertitude, int verbose = 1);
    PredictionResult inverseDensities(const mat &y, int verbose = 1) { return inverseDensitiesOneInversion(y, vec(theta.D, fill::zeros), verbose); };

    Insights getInsights();

private:
    // TODO : retirer L,D,K de GLLiMParameters et le mettre ici.
    // TODO : Google convention: attribute_
    GLLiMConstraints constraints;                               // The constraints of GLLiM model
    GLLiMParameters<TGamma, TSigma> theta;                      // The parameters of the direct GLLiM model
    GLLiMParameters<FullCovariance, FullCovariance> theta_star; // The parameters of the inverse GLLiM model
    Insights insights_;                                         // The relevant data on initialisation and training of GLLiM

    GLLiMParameters<FullCovariance, FullCovariance> inverse(GLLiMParameters<TGamma, TSigma> &theta);
    template <typename TGamma2, typename TSigma2>
    std::tuple<mat, cube, cube> constructGMM(const mat &x, GLLiMParameters<TGamma2, TSigma2> &theta);
    PredictionResult inverseDensitiesOneInversion(const mat &y, const vec &y_incertitude, int verbose = 1);
};

#endif // GLLIM_HPP
