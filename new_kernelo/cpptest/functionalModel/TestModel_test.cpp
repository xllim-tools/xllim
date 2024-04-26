#include <gtest/gtest.h>
#include "../../src/functionalModel/TestModel.hpp"
#include "../../src/dataGeneration/generator/GeneratorFactory.hpp"
#include "../../src/dataGeneration/generator/Generator.hpp"
#include "../../src/dataGeneration/generator/RandomGenerator.hpp"
#include "../../src/dataGeneration/generator/SobolGenerator.hpp"

using namespace DataGeneration;

class TestModelTest : public testing::TestWithParam<std::string>
{
protected:
    TestModelTest()
    {
        model = std::unique_ptr<TestModel>((new TestModel()));
    }
    std::unique_ptr<TestModel> model;
};

TEST_F(TestModelTest, GetLDimension)
{
    ASSERT_EQ(model->getDimensionX(), 4);
}

TEST_F(TestModelTest, GetDDimension)
{
    ASSERT_EQ(model->getDimensionY(), 9);
}

TEST_F(TestModelTest, ToPhysicOnes)
{
    vec x_true(4, fill::ones);
    x_true *= 2;
    vec x(4, fill::ones);
    model->toPhysic(x);
    ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
}

TEST_F(TestModelTest, FromPhysicOnes)
{
    vec x_true(4, fill::ones);
    x_true /= 2;
    vec x(4, fill::ones);
    model->fromPhysic(x);
    ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
}

TEST_F(TestModelTest, FOnZerosX)
{
    vec y_true = {4 + 2 * datum::e, 0.5, datum::e, 3, 0.2, -0.5, -0.2 - datum::e, 2 * datum::e - 1, -0.7};
    y_true *= 0.5;
    vec x(4, fill::zeros);
    vec y(9, fill::ones);
    model->F(x, y);
    ASSERT_TRUE(approx_equal(y_true, y, "reldiff", 1e-8));
}

TEST_P(TestModelTest, GenDataReturnsXYGoodShape)
{
    unsigned N = 10;
    unsigned seed = 12345;
    // vec covariance = vec(9, fill::randu) * 1e-5;
    double noise_ratio = 1e4;
    std::tuple<mat, mat> data = model->genData(N, GetParam(), noise_ratio, seed);
    ASSERT_EQ(std::get<0>(data).n_rows, N) << "X_gen shape (rows) error on " << GetParam();
    ASSERT_EQ(std::get<0>(data).n_cols, 4) << "X_gen shape (cols) error on " << GetParam();
    ASSERT_EQ(std::get<1>(data).n_rows, N) << "Y_gen shape (rows) error on " << GetParam();
    ASSERT_EQ(std::get<1>(data).n_cols, 9) << "Y_gen shape (cols) error on " << GetParam();
}

TEST_P(TestModelTest, YGenApproxEqualFOnXGen)
{
    // This test is based on the validity of FunctionalModel::TestModel::F method
    unsigned N = 50;
    unsigned seed = 12345;
    // vec covariance = vec(9, fill::randu) * 1e-5;
    double noise_ratio = 1e4;
    std::tuple<mat, mat> data = model->genData(N, GetParam(), noise_ratio, seed);
    vec y(9);
    for (unsigned int i = 0; i < N; i++)
    {
        model->F(std::get<0>(data).row(i).t(), y);
        ASSERT_TRUE(approx_equal(std::get<1>(data).row(i).t(), y, "reldiff", 1e-3));
    }
}

INSTANTIATE_TEST_SUITE_P(GenDataSuite,
                         TestModelTest,
                         testing::Values("random", "sobol")); //, "latin_cube"));

TEST_F(TestModelTest, ImportanceSamplingSumWeightsEqualsOne)
{
    // TODO
    const vec weight(2, fill::value(0.5));
    const mat mean(4, 2, fill::value(2));
    const cube covariance(4, 4, 2, fill::value(0.01));
    std::vector<std::tuple<const vec, const mat, const cube>> proposition_gmms;
    proposition_gmms.push_back(std::make_tuple(weight, mean, covariance));
    proposition_gmms.push_back(std::make_tuple(weight, mean, covariance));
    proposition_gmms.push_back(std::make_tuple(weight, mean, covariance));
    mat y(3, 9, fill::value(3));
    mat y_err(3, 9, fill::value(0.003));
    vec y_covariance(9, fill::value(0.06));
    unsigned N_0 = 10;
    unsigned B = 5;
    unsigned J = 8;
    mat results = model->importanceSampling(proposition_gmms, y, y_err, y_covariance, N_0, B, J);
    results.print();
    ASSERT_EQ(results.n_rows, 4); // X dimension
    ASSERT_EQ(results.n_cols, 3); // nb observation
};

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}