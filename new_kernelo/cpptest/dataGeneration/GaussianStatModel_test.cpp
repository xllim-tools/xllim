#include <gtest/gtest.h>
#include "../../src/dataGeneration/statModel/GaussianStatModel.hpp"
#include "../../src/functionalModel/TestModel.hpp"

using namespace DataGeneration;

class GaussianStatModelTest : public testing::TestWithParam<std::string>
{
protected:
    GaussianStatModelTest()
    {
        seed = 12345;
        cov = vec(9, fill::randu) * 1e-5;
        functional_test_model = std::shared_ptr<TestModel>((new TestModel()));
    }

    std::shared_ptr<TestModel> functional_test_model;
    std::shared_ptr<StatModel> stat_model;
    vec cov;
    unsigned int seed;
};

TEST_P(GaussianStatModelTest, GenDataReturnsXYGoodShape)
{
    unsigned int N = 10;
    stat_model = std::shared_ptr<StatModel>(new GaussianStatModel(GetParam(), functional_test_model, cov, seed));
    std::tuple<mat, mat> data = stat_model->gen_data(N);
    ASSERT_EQ(std::get<0>(data).n_rows, N) << "X_gen shape (rows) error on " << GetParam();
    ASSERT_EQ(std::get<0>(data).n_cols, 4) << "X_gen shape (cols) error on " << GetParam();
    ASSERT_EQ(std::get<1>(data).n_rows, N) << "Y_gen shape (rows) error on " << GetParam();
    ASSERT_EQ(std::get<1>(data).n_cols, 9) << "Y_gen shape (cols) error on " << GetParam();
}

TEST_P(GaussianStatModelTest, YGenApproxEqualFOnXGen)
{
    // This test is based on the validity of FunctionalModel::TestModel::F method
    unsigned int N = 50;
    stat_model = std::shared_ptr<StatModel>(new GaussianStatModel(GetParam(), functional_test_model, cov, seed));
    std::tuple<mat, mat> data = stat_model->gen_data(N);
    vec y(9);
    for (unsigned int i = 0; i < N; i++)
    {
        functional_test_model->F(std::get<0>(data).row(i).t(), y);
        ASSERT_TRUE(approx_equal(std::get<1>(data).row(i).t(), y, "reldiff", 1e-3));
    }
}

INSTANTIATE_TEST_SUITE_P(StatModelSuite,
                         GaussianStatModelTest,
                         testing::Values("random", "sobol")); //, "latin_cube"));
