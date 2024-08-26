#include <gtest/gtest.h>
#include <memory>
#include "../../src/functionalModel/TestModel.hpp"
#include "../../src/xllimSolver/gllim.hpp"

// TODO
// developper les tests d'intégration. S'inspirer de oldkernelo/cpptest/predictionTersts/PerformanceTests.cpp

class PerformanceTest : public ::testing::Test
{

protected:
    void SetUp() override
    {
    }

    // std::unique_ptr<FunctionalModel> physical_model;
    // GLLiM<FullCovariance,DiagCovariance> gllim;
};

TEST_F(PerformanceTest, TestModel)
{

    std::unique_ptr<FunctionalModel> physical_model = std::unique_ptr<TestModel>((new TestModel()));
    std::unique_ptr<GLLiM<FullCovariance, DiagCovariance>> gllim = std::unique_ptr<GLLiM<FullCovariance, DiagCovariance>>(new GLLiM<FullCovariance, DiagCovariance>(10, 9, 4, "full", "diag"));

    std::cout << "Generate test data :" << std::endl;
    std::tuple<mat, mat> data = physical_model->genData(1000, "sobol", 0.1, 1234);

    std::cout << "Init GLLiM :" << std::endl;
    unsigned gllim_em_iteration = 10;
    double gllim_em_floor = 1e-12;
    unsigned gmm_kmeans_iteration = 5;
    unsigned gmm_em_iteration = 12;
    double gmm_floor = 1e-12;
    unsigned nb_experiences = 3;
    unsigned seed = 12345;
    int verbose = 1;
    auto start0 = std::chrono::high_resolution_clock::now();
    gllim->initialize(std::get<0>(data).t(), std::get<1>(data).t(), gllim_em_iteration, gllim_em_floor, gmm_kmeans_iteration, gmm_em_iteration, gmm_floor, nb_experiences, seed, verbose);
    auto end0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
    std::cout << "init : " << duration0.count() << std::endl;

    std::cout << "Train GLLiM :" << std::endl;
    auto start1 = std::chrono::high_resolution_clock::now();
    gllim->train(std::get<0>(data).t(), std::get<1>(data).t(), 30, 1e-3, 1e-12, verbose);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "train : " << duration1.count() << std::endl;

    // prediction
    std::cout << "Prediction :" << std::endl;
    mat y_test = mat(std::get<1>(data).rows(1, 50).t());
    auto start2 = std::chrono::high_resolution_clock::now();
    gllim->inverseDensities(y_test);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "prediction : " << duration1.count() << std::endl;

    // Sampling
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
