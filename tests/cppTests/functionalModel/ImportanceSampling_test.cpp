#include <gtest/gtest.h>
#include "../../../src/functionalModel/TestModel.hpp"

class ImportanceSamplingTest : public testing::Test
{
protected:
    ImportanceSamplingTest()
    {
        model = std::unique_ptr<TestModel>((new TestModel()));
        uvec dummy = sort_index(vec(1)); // without this line, the tests below return SEGFAULT ...
        /*
            SUMMARY: AddressSanitizer: SEGV (/path_to_libpython3.10.so) in _PyUnicode_FromId
        */
    }
    void SetUp(const unsigned K, const unsigned N_obs)
    {
        for (size_t i = 0; i < N_obs; ++i)
        {
            weight = vec(K, fill::value(1.0 / K));
            mean = mat(L, K, fill::randu);
            covariance = cube(L, L, K, fill::value(0.01));
            for (size_t i = 0; i < covariance.n_slices; ++i)
            {
                covariance.slice(i) += mat(L, L, fill::eye) * 0.1;
                covariance.slice(i) *= covariance.slice(i).t();
            }
            proposition_gmms.push_back(std::make_tuple(weight, mean, covariance));
        }
        y = mat(N_obs, D, fill::randn);
        y_err = mat(N_obs, D, fill::randu) * 0.001;
        y_covariance = vec(D, fill::randu) * 0.1;
    }
    std::unique_ptr<TestModel> model;
    unsigned L = 4;
    unsigned D = 9;
    vec weight;
    mat mean;
    cube covariance;
    std::vector<std::tuple<vec, mat, cube>> proposition_gmms;
    mat y;
    mat y_err;
    vec y_covariance;
    unsigned N_try = 100;
    unsigned N_experiences = 3;
};

TEST_F(ImportanceSamplingTest, ISReturnsGoodShape)
{
    unsigned N_obs = 20, K = 2, N_0 = 1000;
    SetUp(K, N_obs);
    ImportanceSamplingResult results = model->importanceSampling(proposition_gmms, y, y_err, N_0, 0, 0, y_covariance);
    ASSERT_EQ(results.predictions.n_rows, L);     // X dimension
    ASSERT_EQ(results.predictions.n_cols, N_obs); // nb observation
};

TEST_F(ImportanceSamplingTest, IMISReturnsGoodShape)
{
    unsigned N_obs = 10, K = 5, N_0 = 100, B = 5, J = 8;
    SetUp(K, N_obs);
    ImportanceSamplingResult results = model->importanceSampling(proposition_gmms, y, y_err, N_0, B, J, y_covariance);
    ASSERT_EQ(results.predictions.n_rows, L);     // X dimension
    ASSERT_EQ(results.predictions.n_cols, N_obs); // nb observation
};

TEST_F(ImportanceSamplingTest, Performance)
{
    unsigned N_obs = 20, K = 5;
    SetUp(K, N_obs);

    for (unsigned r = 0; r < N_experiences; r++)
    {
        unsigned N_samples = 100 * (1 + 5 * r);
        unsigned N_0 = N_samples / 10, B = N_samples / 20, J = 18;
        mat x_obs(L, N_obs, fill::randu);
        vec y_obs(D);
        for (unsigned n = 0; n < N_obs; n++)
        {
            model->F(x_obs.col(n), y_obs);
            y.row(n) = y_obs.t();
        }
        double error_on_x_is = 0;
        double error_reconstruction_is = 0;
        ImportanceSamplingResult results = model->importanceSampling(proposition_gmms, y, y_err, N_0, B, J, y_covariance);
        mat x_pred_is = results.predictions;
        vec y_pred_is(D);
        for (unsigned n = 0; n < N_obs; n++)
        {
            model->F(x_pred_is.col(n), y_pred_is);
            error_on_x_is += norm(x_pred_is.col(n) - x_obs.col(n), "inf");
            error_reconstruction_is += norm(y_pred_is - y_obs, 2) / norm(y_obs, 2);
        }
        error_on_x_is /= N_obs;
        error_reconstruction_is /= N_obs;
        EXPECT_TRUE((error_on_x_is >= 0) && (error_on_x_is < 2)) << "Error on x is too high for IS at experience " << r;
        EXPECT_TRUE((error_reconstruction_is >= 0) && (error_reconstruction_is < 1)) << "Reconstruction error is too high for IS at experience " << r;
    }
};

TEST_F(ImportanceSamplingTest, DiagnosticIsOK){
    // TODO
};
