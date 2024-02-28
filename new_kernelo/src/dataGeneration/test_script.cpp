#include <armadillo>
#include "statModel/GaussianStatModel.hpp"
#include "statModel/StatModel.hpp"
#include "generator/Generator.hpp"
#include "generator/RandomGenerator.hpp"
#include "../functionalModel/FunctionalModel.hpp"
#include "../functionalModel/TestModel.hpp"

int main(){
    // Stat model
    std::shared_ptr<Functional::TestModel> testModel = std::shared_ptr<Functional::TestModel>((new TestModel()));
    std::shared_ptr<Functional::FunctionalModel> functionalModel = std::shared_ptr<Functional::FunctionalModel>(testModel);
    // BasicDependentGaussianStatModel = std::shared_ptr<DataGeneration::StatModel>(DataGeneration::DependentGaussianStatModelConfig("sobol", functionalModel, 50 , 12345).create());
    arma::vec cov(functionalModel->get_D_dimension(), fill::ones);
    std::cout<< "hey"<<std::endl;
    std::shared_ptr<DataGeneration::StatModel> BasicGaussianStatModel =  std::shared_ptr<DataGeneration::StatModel>(new DataGeneration::GaussianStatModel("sobol", functionalModel, cov, 12345));
    
    std::cout << "Generate test data :" << std::endl;
    std::tuple<mat, mat> data = BasicGaussianStatModel->gen_data(10);
    std::cout<< "FINI"<<std::endl;
    std::get<0>(data).print("X");
    std::cout<< "hey"<<std::endl;
    std::get<1>(data).print("Y");
    // std::shared_ptr<DataGeneration::StatModel> BasicGaussianStatModel = std::shared_ptr<DataGeneration::StatModel>(DataGeneration::GaussianStatModel("sobol", functionalModel, cov, 12345));
    // Note: In the IMIS process only F(x) is evaluated within density_X_Y() function. generator type and seed have no impact here
    // We should test with the other statModel: GaussianStatModel
    std::cout<< "OOOO"<<std::endl;
    return 0;
}