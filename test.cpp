#include <gtest/gtest.h>
#include "src/functionalModel/FunctionalModel.h"
#include "src/dataGeneration/StatModel.h"
#include "src/dataGeneration/creators.h"
#include "src/learningModel/covariances/Icovariance.h"
#include "src/learningModel/initializers/MultInitializer.h"
#include "src/learningModel/initializers/FixedInitializer.h"
#include "src/learningModel/estimators/EmEstimator.h"
#include "src/learningModel/gllim/IGLLiMLearning.h"
#include "src/learningModel/gllim/GLLiMLearning.h"
#include "src/prediction/Predictor.h"
#include "src/prediction/PredictionResult.h"

#include "src/functionalModel/HapkeModel/HapkeVersions/Hapke02Model.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/SixParamsModel.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/FourParamsModel.h"
#include "src/importanceSampling/ImportanceSampler.h"
#include "src/importanceSampling/target/ISTargetDependent.h"
#include "src/importanceSampling/proposition/GaussianMixtureProposition.h"
#include "src/importanceSampling/proposition/GaussianRegularizedProposition.h"
#include <gtest/gtest.h>
#include <armadillo>
#include <iostream>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using namespace learningModel;
using namespace Functional;
using namespace DataGeneration;
namespace pt = boost::property_tree;


template <typename T , typename U >
struct ExperienceConfig{
    unsigned N_train;
    unsigned N_test;
    vec var_obs;
    unsigned K;
    unsigned K_merged;
    double threshold;
    unsigned D;
    unsigned L;
    double norm;
    unsigned r_noise;
    bool is = false;
    std::shared_ptr<FunctionalModel> functionalModel;
    std::shared_ptr<StatModel> statModel;
    std::shared_ptr<GLLiMLearning<T,U>> gllim;
};

struct ExperienceResult{
    double F_mean = 0;
    double Me_mean = 0;
    double Ce_mean = 0;
    double Yme_mean = 0;
    double Yb_mean = 0;
    double Yce_mean = 0;
    double F_median = 0;
    double Me_median = 0;
    double Ce_median = 0;
    double Yme_median = 0;
    double Yb_median = 0;
    double Yce_median = 0;
    double V1 = 0;
    double V2 = 0;
};

class ExpoFunctional : public FunctionalModel{
public:
    int L, D;

    ExpoFunctional(int L, int D){
        this->L = L;
        this->D = D;
    }

    int get_D_dimension() override {
        return D;
    }

    int get_L_dimension() override {
        return L;
    }

    void to_physic(rowvec &x) override {
        x = x * 3 - 1;
    }

    void to_physic(double *x, unsigned int size) override {
    }

    void from_physic(double *x, unsigned int size) override {
    }

    void F(rowvec x, rowvec &y) override {
        to_physic(x);
        y = exp(x);
    }
};

template <typename T , typename U >
ExperienceResult executeExperience(ExperienceConfig<T,U> config){
    ExperienceResult result;
    
    auto start0 = std::chrono::high_resolution_clock::now();
    auto start1 = std::chrono::high_resolution_clock::now();
    auto start2 = std::chrono::high_resolution_clock::now();
    auto start3 = std::chrono::high_resolution_clock::now();
    auto start4 = std::chrono::high_resolution_clock::now();

    auto end0 = std::chrono::high_resolution_clock::now();
    auto end1 = std::chrono::high_resolution_clock::now();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto end3 = std::chrono::high_resolution_clock::now();


    auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);


    std::cout << "Generate test data :" << std::endl;
    std::tuple<mat, mat> data = config.statModel->gen_data(config.N_train);
    rowvec y_temp_1(config.D);

    mat x_test(config.N_test,config.L, fill::ones);
    x_test *= 0.5;
    mat y_test(config.N_test, config.D);

    for(unsigned n=0; n<config.N_test; n++){

        config.functionalModel->F(x_test.row(n), y_temp_1);
        y_test.row(n) = y_temp_1;
    }

    std::cout << "Init GLLiM :" << std::endl;
    start0 = std::chrono::high_resolution_clock::now();
    config.gllim->initialize(std::get<0>(data),std::get<1>(data));
    end0 = std::chrono::high_resolution_clock::now();
    duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);

    std::cout << "init : " << duration0.count() << std::endl;


    std::cout << "Train GLLiM :" << std::endl;
    start1 = std::chrono::high_resolution_clock::now();
    config.gllim->train(std::get<0>(data),std::get<1>(data));
    end1 = std::chrono::high_resolution_clock::now();
    duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    std::cout << "train : " << duration1.count() << std::endl;

    result.F_mean = 0;
    result.Me_mean = 0;
    result.Ce_mean = 0;
    result.Yme_mean = 0;
    result.Yce_mean = 0;
    result.Yb_mean = 0;
    result.F_median = 0;
    result.Me_median = 0;
    result.Ce_median = 0;
    result.Yme_median = 0;
    result.Yce_median = 0;
    result.Yb_median = 0;
    result.V1 = 0;
    result.V2 = 0;

    prediction::Predictor predictorByCenters(std::dynamic_pointer_cast<IGLLiMLearning>(config.gllim), config.K_merged, 10, config.threshold);

    vec cov_is(config.D, fill::ones);
    vec estimation(config.D,fill::zeros);

    duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(start0 - start0);
    duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(start1 - start1);

    std::cout << "Predict :" << std::endl;
    importanceSampling::ISTarget target;
    target.setTarget(config.statModel);
    importanceSampling::ImportanceSampler sampler(1000, std::make_shared<importanceSampling::ISTarget>(target));


    cube allPred(config.L, config.K_merged, config.N_test);
    for(unsigned n=0; n<config.N_test; n++){
        start2 = std::chrono::high_resolution_clock::now();
        prediction::PredictionResult predics = predictorByCenters.predict(y_test.row(n).t(), config.var_obs);
        end2 = std::chrono::high_resolution_clock::now();
        duration2 += std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

        //Mean
        arma::gmm_full gmm = config.gllim->logDensity(config.gllim->getParameters(), x_test.row(n).t());
        estimation.fill(0);
        for(unsigned k=0; k<gmm.hefts.n_cols; k++){
            estimation += gmm.hefts(k) * gmm.means.col(k);
        }
        unsigned cpt1 = 0;
        for(auto element : predics.meanPredResult.mean){
            if(element <= 1 && element >= 0)
                cpt1 ++;
        }
        if(cpt1 == config.L){
            if(config.is){
                start1 = std::chrono::high_resolution_clock::now();

                importanceSampling::GaussianMixtureProposition prop(predics.meanPredResult.gmm_weights,
                        predics.meanPredResult.gmm_means,
                        predics.meanPredResult.gmm_covs);
                importanceSampling::ISResult res = sampler.execute(
                        std::make_shared<importanceSampling::GaussianMixtureProposition>(prop),
                        y_test.row(n).t(),
                        cov_is / 1000);

                config.functionalModel->F(res.mean.t(), y_temp_1);
                end1 = std::chrono::high_resolution_clock::now();
                duration1 += std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

                result.F_mean += max(abs((estimation - y_test.row(n))));//config.norm);
                result.Me_mean += max(abs((res.mean - x_test.row(n).t())));
                result.Yme_mean += max(abs((y_temp_1 - y_test.row(n))));//config.norm));
            }else{
                config.functionalModel->F(predics.meanPredResult.mean.t(), y_temp_1);
                result.F_mean += max(abs((estimation - y_test.row(n))));
                result.Me_mean += max(abs((predics.meanPredResult.mean - x_test.row(n).t())));
                result.Yme_mean += max(abs((y_temp_1 - y_test.row(n))));
            }
            result.V1 += 1;
        }

        //Centers
        double Ce = datum::inf, Yb = datum::inf;
        double V2 = 0;
        for(unsigned k=0; k<config.K_merged; k++){
            //allPred.slice(n).col(k) = predics[k].first;
            unsigned cpt2 = 0;
            for(auto element : predics.centerPredResult.means.col(k)){
                if(element <= 1 && element >= 0)
                    cpt2 ++;
            }
            if(cpt2 == config.L){
                V2 += 1;
                if(config.is){
                    start3 = std::chrono::high_resolution_clock::now();
                    vec center = predics.centerPredResult.means.col(k);
                    importanceSampling::GaussianRegularizedProposition prop(center,
                                                                            predics.centerPredResult.covs.slice(k));
                    importanceSampling::ISResult res = sampler.execute(
                            std::make_shared<importanceSampling::GaussianRegularizedProposition>(prop),
                            y_test.row(n).t(),
                            cov_is / 1000);
                    config.functionalModel->F(res.mean.t(), y_temp_1);
                    end3 = std::chrono::high_resolution_clock::now();
                    duration3 += std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
                    result.Yce_mean += max(abs((y_temp_1 - y_test.row(n)))) / config.K_merged;
                    Ce = std::min(Ce, max(abs((res.mean- x_test.row(n).t()))));
                    Yb = std::min(Yb, max(abs((y_temp_1 - y_test.row(n)))));
                }else{

                    config.functionalModel->F(predics.centerPredResult.means.col(k).t(), y_temp_1);
                    result.Yce_mean += max(abs((y_temp_1 - y_test.row(n)))) / config.K_merged;
                    Ce = std::min(Ce, max(abs((predics.centerPredResult.means.col(k) - x_test.row(n).t()))));
                    Yb = std::min(Yb, max(abs((y_temp_1 - y_test.row(n)))));
                }
            }else{
                Ce = 0;
                Yb = 0;
            }
        }
        result.V2 += V2/config.K_merged;
        if(Ce != -datum::inf && Yb != -datum::inf){
            result.Ce_mean += Ce;
            result.Yb_mean += Yb;
        }
    }


    /*Mat<unsigned> regu = prediction::Predictor::regularize(allPred);
    cube allPred_reg(config.L, config.K_merged, config.N_test);
    for(unsigned i=0; i<config.N_test; i++){
        for(unsigned j=0 ; j<config.K_merged; j++){
            allPred_reg.slice(i).col(j) = allPred.slice(i).col(regu(i,j));
        }
    }*/

    /*allPred.print("all Pred");
    allPred_reg.print(" all Pred regu");*/

    auto end4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4);

    std::cout << "pred mean : " << duration0.count() << std::endl;
    std::cout << "pred center : " << duration2.count() << std::endl;
    std::cout << "is mean : " << duration1.count() << std::endl;
    std::cout << "is center : " << duration3.count() << std::endl;
    std::cout << "all : " << duration4.count() << std::endl;

    result.F_mean /= config.N_test;
    result.Me_mean /= config.N_test;
    result.Ce_mean /= config.N_test;
    result.Yme_mean /= config.N_test;
    result.Yce_mean /= config.N_test;
    result.Yb_mean /= config.N_test;
    result.V1 /= config.N_test;
    result.V2 /= config.N_test;

    return result;
};

class ExpoFunctionalTest : public ::testing::Test{
    protected:
        void SetUp() override {
        };
    };



TEST_F(ExpoFunctionalTest, ExpoFunctionalTest_test){
    std::shared_ptr<MultInitConfig> myConfig (
        new MultInitConfig(
                123456789,
                5,
                3,
                std::make_shared<GMMLearningConfig>(GMMLearningConfig(10,5,1e-08)),
                std::make_shared<EMLearningConfig>(EMLearningConfig(10,5,1e-08))));
    MultInitializer<FullCovariance,DiagCovariance> initializer(myConfig);

    std::shared_ptr<FixedInitConfig> fixedConfig (
            new FixedInitConfig(
                    123456789,
                    std::make_shared<GMMLearningConfig>(GMMLearningConfig(0,5,1e-08)),
                    std::make_shared<EMLearningConfig>(EMLearningConfig(4,0,1e-08))
            ));

    std::shared_ptr<EMLearningConfig> myLearningconfig (new EMLearningConfig(30,2.0,1e-08));
    EmEstimator<FullCovariance, DiagCovariance> estimator(myLearningconfig);

    std::shared_ptr<GLLiMLearning<FullCovariance,DiagCovariance>> gllim;
    gllim = std::shared_ptr<GLLiMLearning<FullCovariance,DiagCovariance>>(
            new GLLiMLearning<FullCovariance,DiagCovariance>(
                    std::make_shared<MultInitializer<FullCovariance,DiagCovariance>>(initializer),
                    std::make_shared<EmEstimator<FullCovariance, DiagCovariance>>(estimator),100));

    ExperienceConfig<FullCovariance, DiagCovariance> config;
    config.norm = 0.25;
    config.K = 100;
    config.K_merged = 2;
    config.r_noise = 1000000;
    config.N_train = 1000;
    config.threshold = 0.01;
    config.gllim = gllim;
    config.N_test = 10;
    config.is = true;
    for(unsigned i=1; i<2; i+=2){
        config.L = i;
        config.D = i;
        config.var_obs = vec(config.D, fill::zeros);
        config.functionalModel = std::shared_ptr<FunctionalModel>(new ExpoFunctional(config.L, config.D));
        config.statModel = std::shared_ptr<StatModel>(DependentGaussianStatModelConfig("sobol", config.functionalModel,config.r_noise , 12345).create());

        ExperienceResult result = executeExperience(config);
        std::cout << "Experience : D , L : "<< i << std::endl;
        std::cout << result.F_mean * 100 << std::endl;
        std::cout << result.Me_mean * 100 << std::endl;
        std::cout << result.Ce_mean * 100 << std::endl;
        std::cout << result.Yme_mean * 100 << std::endl;
        std::cout << result.Yce_mean * 100 << std::endl;
        std::cout << result.Yb_mean * 100 << std::endl;
        std::cout << result.V1 * 100 << std::endl;
        std::cout << result.V2 * 100 << std::endl;
    }
};

int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    // testing::GTEST_FLAG(filter) = "*HapkeFunctionalTest*";
    return RUN_ALL_TESTS();
};
