#include <armadillo>
#include <iostream>

#include "src/importanceSampling/creators.h"
// #include "src/importanceSampling/Imis.h"
#include "src/importanceSampling/target/ISTarget.h"
#include "src/importanceSampling/proposition/ISProposition.h"
#include "src/importanceSampling/ISResult.h"
#include "src/dataGeneration/StatModel.h"
#include "src/dataGeneration/creators.h"
#include "src/functionalModel/creators.h"
#include "src/functionalModel/TestModel/TestModel.h"

// #include <gtest/gtest.h>
// #include "src/functionalModel/FunctionalModel.h"
// #include "src/dataGeneration/StatModel.h"
// #include "src/dataGeneration/creators.h"
// #include "src/learningModel/covariances/Icovariance.h"
// #include "src/learningModel/initializers/MultInitializer.h"
// #include "src/learningModel/initializers/FixedInitializer.h"
// #include "src/learningModel/estimators/EmEstimator.h"
// #include "src/learningModel/gllim/IGLLiMLearning.h"
// #include "src/learningModel/gllim/GLLiMLearning.h"
// #include "src/prediction/Predictor.h"
// #include "src/prediction/PredictionResult.h"

// #include "src/functionalModel/HapkeModel/HapkeVersions/Hapke02Model.h"
// #include "src/functionalModel/HapkeModel/HapkeAdapters/SixParamsModel.h"
// #include "src/functionalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.h"
// #include "src/functionalModel/HapkeModel/HapkeAdapters/FourParamsModel.h"
// #include "src/importanceSampling/ImportanceSampler.h"
// #include "src/importanceSampling/target/ISTargetDependent.h"
// #include "src/importanceSampling/proposition/GaussianMixtureProposition.h"
// #include "src/importanceSampling/proposition/GaussianRegularizedProposition.h"
// #include <gtest/gtest.h>
// #include <armadillo>
// #include <iostream>
// #include <boost/property_tree/json_parser.hpp>
// #include <boost/property_tree/ptree.hpp>



using namespace std;
using namespace importanceSampling;
using namespace arma;

int main(){

    // ------------- INPUTS -------------

    // constructor
    unsigned N_0(100), B(10), J(180);
    std::shared_ptr<ISTarget> isTarget;
    std::shared_ptr<DataGeneration::StatModel> statModel;
    std::shared_ptr<Functional::TestModel> myModel = std::shared_ptr<Functional::TestModel>((new TestModel()));
    unsigned L = myModel->get_L_dimension();
    unsigned D = myModel->get_D_dimension();
    double r_noise(50);
    std::shared_ptr<Functional::FunctionalModel> functionalModel = std::shared_ptr<Functional::FunctionalModel>(myModel);
    statModel = std::shared_ptr<DataGeneration::StatModel>(DataGeneration::DependentGaussianStatModelConfig("sobol", functionalModel, r_noise , 12345).create());

    // execute
    vec y_cov(D, arma::fill::zeros);
    vec x_obs = vec("0.1 0.2 0.5 0.8");
    rowvec y_obs_rowvec(D);
    myModel->F(x_obs.t(), y_obs_rowvec);
    vec y_obs = y_obs_rowvec.t();
    // y_obs = vec("4.0 0.0 4.5 2.5 0.0 0.0 -1.0 0.0 -0.5");
    // y_cov *= 0.1;
    // double *y_obs_double, *y_cov_double;
    // unsigned size(1);

    std::shared_ptr<ISProposition> isProposition;
    

    // ------------- EXECUTE -------------

    // create
    // std::shared_ptr<Imis> imis_sampler = ImisConfig(N_0,B,J,statModel).create();
    importanceSampling::ISTarget target;
    target.setTarget(statModel);
    importanceSampling::Imis imis_sampler(N_0,B,J, std::make_shared<importanceSampling::ISTarget>(target));

    // ISResult execute(std::shared_ptr<ISProposition> isProposition, const vec &y_obs, const vec &y_cov);
    // importanceSampling::ISResult isResult = imis_sampler->execute(isProposition, y_obs, y_cov);
    vec gmm_weights(1, arma::fill::ones);
    mat gmm_means(L, 1, arma::fill::zeros);
    // Note: there is no "to_physic" fonction in TestModel that is why x sampled must be in [0,1]. So the prop gaussian must correct this
    // gmm_means = mat("0.5 ; 0.5 ; 0.4 ; 0.6");
    cube gmm_covs(L, L, 1, arma::fill::zeros); // Faire un truc symetrique !!
    gmm_covs.slice(0) = {   {1, 0, 0, 0},
                            {0, 1, 0, 0},
                            {0, 0, 1, 0},
                            {0, 0, 0, 1},
                        };
    gmm_covs *= 0.001;
    importanceSampling::GaussianMixtureProposition prop(
                        gmm_weights,
                        gmm_means,
                        gmm_covs);
    importanceSampling::ISResult res = imis_sampler.execute(
                        std::make_shared<importanceSampling::GaussianMixtureProposition>(prop),
                        y_obs,
                        y_cov);

    // void execute(std::shared_ptr<ISProposition> isProposition, double *y_obs, double *y_cov, unsigned size, std::shared_ptr<ImportanceSamplingResult> resultExport);
    // std::shared_ptr<importanceSampling::ImportanceSamplingResult> importanceSamplingResult;
    // imis_sampler->execute(isProposition, y_obs_double, y_cov_double, size, importanceSamplingResult);





    // ------------- RESULTS ANALYSIS -------------
    vec res_mean = res.mean;
    vec res_cov = res.covariance;
    int nb_effective_sample = res.diagnostic.nb_effective_sample;
    double effective_sample_size = res.diagnostic.effective_sample_size;
    double qn = res.diagnostic.qn;
    std::cout << res_mean << std::endl;
    std::cout << res_cov << std::endl;
    std::cout << nb_effective_sample << std::endl;
    std::cout << effective_sample_size << std::endl;
    std::cout << qn << std::endl;
    double err = arma::norm(res_mean-x_obs, "inf");
    std::cout << x_obs << std::endl;
    std::cout << err << std::endl;




    // FROM performancceTests.cpp


    // for(unsigned n=0; n<N_test; n++){
    //     prediction::PredictionResult predics = predictorByCenters.predict(y_test.row(n).t(), config.var_obs);
    //     importanceSampling::GaussianMixtureProposition prop(predics.meanPredResult.gmm_weights,
    //             predics.meanPredResult.gmm_means,
    //             predics.meanPredResult.gmm_covs);
    //     importanceSampling::ISResult res = sampler.execute(
    //             std::make_shared<importanceSampling::GaussianMixtureProposition>(prop),
    //             y_test.row(n).t(),
    //             cov_is / 1000);

    //     functionalModel->F(res.mean.t(), y_temp_1);

    //     result.F_mean += max(abs((estimation - y_test.row(n))));//config.norm);
    //     result.Me_mean += max(abs((res.mean - x_test.row(n).t())));
    //     result.Yme_mean += max(abs((y_temp_1 - y_test.row(n))));//config.norm));
    // }

    return 0;
}



