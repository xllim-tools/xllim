#include <gtest/gtest.h>
#include "../../../src/functionalModel/TestModel.hpp"
#include "../../../src/generator/GeneratorFactory.hpp"
// #include "../../../src/dataGeneration/generator/Generator.hpp"
// #include "../../../src/dataGeneration/generator/RandomGenerator.hpp"
// #include "../../../src/dataGeneration/generator/SobolGenerator.hpp"

using namespace DataGeneration;

class DataGenerationTest : public testing::TestWithParam<std::string>
{
protected:
    DataGenerationTest()
    {
        model = std::unique_ptr<TestModel>((new TestModel()));
    }
    std::unique_ptr<TestModel> model;
};

TEST_P(DataGenerationTest, GenDataReturnsXYGoodShape)
{
    unsigned N = 10;
    unsigned seed = 12345;
    // vec covariance = vec(9, fill::randu) * 1e-5;
    double noise_ratio = 0.01;
    std::tuple<mat, mat> data = model->genData(N, GetParam(), noise_ratio, seed);
    ASSERT_EQ(std::get<0>(data).n_rows, 4) << "X_gen shape (rows) error on " << GetParam();
    ASSERT_EQ(std::get<0>(data).n_cols, N) << "X_gen shape (cols) error on " << GetParam();
    ASSERT_EQ(std::get<1>(data).n_rows, 9) << "Y_gen shape (rows) error on " << GetParam();
    ASSERT_EQ(std::get<1>(data).n_cols, N) << "Y_gen shape (cols) error on " << GetParam();
}

TEST_P(DataGenerationTest, YGenApproxEqualFOnXGen)
{
    // This test is based on the validity of FunctionalModel::TestModel::F method
    unsigned N = 50;
    unsigned seed = 12345;
    // vec covariance = vec(9, fill::randu) * 1e-5;
    double noise_ratio = 0.01;
    std::tuple<mat, mat> data = model->genData(N, GetParam(), noise_ratio, seed);
    vec y(9);
    for (unsigned int i = 0; i < N; i++)
    {
        model->F(std::get<0>(data).col(i), y);
        ASSERT_TRUE(approx_equal(std::get<1>(data).col(i), y, "reldiff", 1e-3));
    }
}

INSTANTIATE_TEST_SUITE_P(GenDataSuite,
                         DataGenerationTest,
                         testing::Values("random", "sobol")); //, "latin_cube"));
