#include <armadillo>
#include <utility>
#include "omp.h"

#include "src/learningModel/configs/InitConfig.h"
#include "src/learningModel/gllim//GLLiMLearning.h"
#include "src/learningModel/initializers/FixedInitializer.h"
#include "src/learningModel/initializers/MultInitializer.h"
#include "src/learningModel/gllim/GLLiMParameters.h"
#include "src/learningModel/configs/LearningConfig.h"
#include "src/learningModel/estimators/GmmEstimator.h"
#include "src/learningModel/estimators/EmEstimator.h"
#include "src/learningModel/covariances/Icovariance.h"
#include "src/dataGeneration/SobolGenerator.h"
#include "src/dataGeneration/RandomGenerator.h"
#include "src/prediction/Predictor.h"
#include "src/functionalModel/FunctionalModel.h"
#include "src/functionalModel/HapkeModel/HapkeVersions/Hapke02Model.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/FourParamsModel.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/SixParamsModel.h"
#include "src/dataGeneration/LatinCubeGenerator.h"
#include "src/dataGeneration/creators.h"


#include <iostream>
#include <cstring>
#include <exception>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <fstream>


using namespace std;
namespace pt = boost::property_tree;

using namespace learningModel;
using namespace arma;

int main(){

    // Data Generation
    auto *geometries = new double[50*3];
    unsigned i = 0;


    pt::ptree root;
    pt::read_json("../test_hapke.json", root);  // Load the json file in this ptree
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("eme"))
    {
        geometries[i*3+0] = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("inc"))
    {
        geometries[i*3+1] = stod(v.second.data());

        i++;
    }
    i = 0;



    for (pt::ptree::value_type& v : root.get_child("phi"))
    {
        geometries[i*3+2] = stod(v.second.data());
        i+=1;
    }



    mat photometries = mat(10000,6);

    i = 0;
    for (pt::ptree::value_type& v : root.get_child("omega"))
    {
        photometries(i,0) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("b"))
    {
        photometries(i,2) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("c"))
    {
        photometries(i,3) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("hh"))
    {
        photometries(i,5) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("b0"))
    {
        photometries(i,4) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("theta0"))
    {
        photometries(i,1) = stod(v.second.data()) / 30.0;
        i++;
    }


    mat y = mat(10000,50);
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("y"))
    {
        int j = 0;
        for(pt::ptree::value_type& elem : v.second){
            y(i,j) = elem.second.get_value<double>();
            j++;
        }
        i++;
    }



    y += (mat(y.n_rows, y.n_cols, fill::randn) * 1/100);




    // testing initialization
    std::shared_ptr<MultInitConfig> myConfig (
            new MultInitConfig(
                    123456789,
                    3,
                    1,
                    make_shared<GMMLearningConfig>(GMMLearningConfig(0,4,1e-10)),
                    make_shared<EMLearningConfig>(EMLearningConfig(3,0,1e-08))));

    MultInitializer<FullCovariance,FullCovariance> initializer(myConfig);
    std::shared_ptr<GLLiMParameters <FullCovariance, FullCovariance>> gllim_initialized = initializer.execute(photometries.submat(0,0,N-1,L-1), y.submat(0,3,N-1,49),K);

    std::shared_ptr<GMMLearningConfig> myLearningconfig (new GMMLearningConfig(0,5,0.00000001);
    GmmEstimator estimator (myLearningconfig);

    // std::shared_ptr<GMMLearningConfig> myLearningconfig (new GMMLearningConfig(0,5,0.00000001));
    // GmmEstimator estimator (myLearningconfig);
    // EmEstimator<FullCovariance, DiagCovariance> estimator(myLearningconfig);

    auto start = chrono::high_resolution_clock::now();

    estimator.execute(photometries.submat(0,0,N-1,L-1), y.submat(0,3,N-1,49), gllim_initialized);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << duration.count() << endl;

    std::shared_ptr<IGLLiMLearning> gllim (
            new GLLiMLearning<FullCovariance,FullCovariance>(
                    make_shared<MultInitializer<FullCovariance,FullCovariance>>(initializer),
                    make_shared<GmmEstimator>(estimator),50));

    std::shared_ptr<Functional::FunctionalModel> myModel (new Hapke02Model(geometries, 50, 3, std::shared_ptr<HapkeAdapter>(new SixParamsModel()), 30.0));
    std::shared_ptr<DataGeneration::StatModel> statModel = DataGeneration::DependentGaussianStatModelConfig("sobol", myModel,20, 123456789).create();

    std::tuple<mat, mat> gen = statModel->gen_data(10000);

    std::cout << std::get<1>(gen).max() << std::endl;

    gllim->initialize(std::get<0>(gen), std::get<1>(gen));
    gllim->train(std::get<0>(gen),std::get<1>(gen));
    vec cov_obs(47, fill::randu);
    vec y_obs = y.row(5).subvec(3,49).t();

    prediction::Predictor predictor(gllim, 2, 1e-10);

    std::vector<std::pair<vec,vec>> centers;

    for(unsigned n=0; n<1; n++){
        centers = predictor.predict(y_obs, cov_obs);
        for(const auto center : centers){
            photometries.row(5).print();
            center.first.t().print();
            std::cout << max(arma::abs(center.first.t() - photometries.row(5))) << std::endl;
        }
    }




    // Functional::TestModel myModel;
    // DataGeneration::DependentGaussianStatModel myStatModel("sobol", myModel, 10000, 12345);
    // unsigned N_0(10);
    // unsigned B(10);
    // unsigned J(5);
    // importanceSampling::ISTarget isTarget;
    // isTarget.setTarget(myStatModel);
    // importanceSampling::Imis imis(N_0,B,J,isTarget);
    // importanceSampling::GaussianMixtureProposition proposition(vec &weights, mat &means, cube &covariances)
    // y_test;
    // cov_is;
    // imis.execute(proposition, y_test, cov_is);

    return 0;
}