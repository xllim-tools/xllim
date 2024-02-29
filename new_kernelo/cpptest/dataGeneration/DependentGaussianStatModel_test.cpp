#include <gtest/gtest.h>
#include "../../src/dataGeneration/statModel/DependentGaussianStatModel.hpp"
#include "../../src/functionalModel/TestModel.hpp"

using namespace DataGeneration;

class DependentGaussianStatModelTest : public testing::TestWithParam<std::string>
{
protected:
    DependentGaussianStatModelTest()
    {
        seed = 12345;
        r = 10000;
        functional_test_model = std::shared_ptr<TestModel>((new TestModel()));
    }

    std::shared_ptr<TestModel> functional_test_model;
    std::shared_ptr<StatModel> stat_model;
    double r;
    unsigned int seed;
};

TEST_P(DependentGaussianStatModelTest, GenDataReturnsXYGoodShape)
{
    unsigned int N = 10;
    stat_model = std::shared_ptr<StatModel>(new DependentGaussianStatModel(GetParam(), functional_test_model, r, seed));
    std::tuple<mat, mat> data = stat_model->gen_data(N);
    ASSERT_EQ(std::get<0>(data).n_rows, N) << "X_gen shape (rows) error on " << GetParam();
    ASSERT_EQ(std::get<0>(data).n_cols, 4) << "X_gen shape (cols) error on " << GetParam();
    ASSERT_EQ(std::get<1>(data).n_rows, N) << "Y_gen shape (rows) error on " << GetParam();
    ASSERT_EQ(std::get<1>(data).n_cols, 9) << "Y_gen shape (cols) error on " << GetParam();
}

TEST_P(DependentGaussianStatModelTest, YGenApproxEqualFOnXGen)
{
    std::cout << std::to_string(r) << std::endl;
    // This test is based on the validity of FunctionalModel::TestModel::F method
    unsigned int N = 50;
    stat_model = std::shared_ptr<StatModel>(new DependentGaussianStatModel(GetParam(), functional_test_model, r, seed));
    std::tuple<mat, mat> data = stat_model->gen_data(N);
    vec y(9);
    for (unsigned int i = 0; i < N; i++)
    {
        functional_test_model->F(std::get<0>(data).row(i).t(), y);
        ASSERT_TRUE(approx_equal(std::get<1>(data).row(i).t(), y, "reldiff", 1e-3));
    }
}

INSTANTIATE_TEST_SUITE_P(StatModelSuite,
                         DependentGaussianStatModelTest,
                         testing::Values("random", "sobol")); //, "latin_cube"));
