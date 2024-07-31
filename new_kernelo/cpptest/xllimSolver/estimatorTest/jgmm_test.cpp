#include <gtest/gtest.h>
#include "../../../src/xllimSolver/jgmm.hpp"

class JGMMTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        unsigned L = 2, D = 5, K = 2;

        mat A_k = mat({{2, 0},
                       {0, 1},
                       {6, 0},
                       {3, 2},
                       {5, -1}});
        vec B_k = vec({3, 2, 6, 1, 3});
        vec C_k = vec({0, 3});
        vec Pi = vec({0.8, 0.2});
        mat sigma_k = mat(D, D, fill::ones);
        sigma_k *= 0.1;
        mat gamma_k = mat(L, L, fill::ones);
        gamma_k *= 0.1;

        theta = GLLiMParameters<FullCovariance, FullCovariance>(L, D, K);
        theta.Pi = Pi;
        for (unsigned k = 0; k < K; k++)
        {
            theta.Sigma[k] = FullCovariance(sigma_k);
            theta.Gamma[k] = FullCovariance(gamma_k);
            theta.A.slice(k) = A_k;
            theta.B.col(k) = B_k;
            theta.C.col(k) = C_k;
        }

        expected_weights = vec({0.8, 0.2});
        expected_means = mat({{0, 0},
                              {3, 3},
                              {3, 3},
                              {5, 5},
                              {6, 6},
                              {7, 7},
                              {0, 0}});
        expected_covs = cube(7, 7, 2);
        for (unsigned k = 0; k < 2; k++)
        {
            expected_covs.slice(k) = mat({
                {0.1, 0.1, 0.2, 0.1, 0.6, 0.5, 0.4},
                {0.1, 0.1, 0.2, 0.1, 0.6, 0.5, 0.4},
                {0.2, 0.2, 0.5, 0.3, 1.3, 1.1, 0.9},
                {0.1, 0.1, 0.3, 0.2, 0.7, 0.6, 0.5},
                {0.6, 0.6, 1.3, 0.7, 3.7, 3.1, 2.5},
                {0.5, 0.5, 1.1, 0.6, 3.1, 2.6, 2.1},
                {0.4, 0.4, 0.9, 0.5, 2.5, 2.1, 1.7},

            });
        }
    };

    JGMM estimator;
    GLLiMParameters<FullCovariance, FullCovariance> theta;
    rowvec expected_weights;
    mat expected_means;
    cube expected_covs;
};

TEST_F(JGMMTest, GLLiMParameterstoJGMM)
{
    // TODO friend class ?
    // estimator.GLLiMParameterstoJGMM(theta);

    // ASSERT_EQ(accu(expected_weights != estimator.jgmm.hefts), 0);
    // ASSERT_EQ(accu(expected_means != estimator.jgmm.means), 0);
    // ASSERT_EQ(accu(expected_covs - estimator.jgmm.fcovs), 0);
}

TEST_F(JGMMTest, JGMMtoGLLiMParameters)
{
    // TODO friend class ?
    // estimator.jgmm.set_params(expected_means, expected_covs, expected_weights);
    // GLLiMParameters<FullCovariance, FullCovariance> calulated_theta = estimator.JGMMtoGLLiMParameters();
    // ASSERT_EQ(accu(calulated_theta.Pi - theta.Pi), 0);
    // ASSERT_TRUE(approx_equal(calulated_theta.A, theta.A, "absdiff", 1e-15));
    // ASSERT_TRUE(approx_equal(calulated_theta.B, theta.B, "absdiff", 1e-15));
    // ASSERT_TRUE(approx_equal(calulated_theta.C, theta.C, "absdiff", 1e-15));

    // for (unsigned k = 0; k < 2; k++)
    // {
    //     ASSERT_TRUE(approx_equal(calulated_theta.Gamma[k].get_mat(), theta.Gamma[k].get_mat(), "absdiff", 1e-15));
    //     ASSERT_TRUE(approx_equal(calulated_theta.Sigma[k].get_mat(), theta.Sigma[k].get_mat(), "absdiff", 1e-15));
    // }
}

TEST_F(JGMMTest, trainSmall)
{
    // TODO : find a reproductible example : small dataset
    // mat x;
    // mat y;
    // unsigned kmeans_iteration;
    // unsigned em_iteration;
    // double floor;
    // GLLiMParameters<FullCovariance,FullCovariance> theta_expected;
    // GLLiMParameters<FullCovariance,FullCovariance> theta_trained = train(x, y, theta, kmeans_iteration, em_iteration, floor);
}

TEST_F(JGMMTest, trainLarge)
{
    // TODO : find a reproductible example : large dataset
    // mat x;
    // mat y;
    // unsigned kmeans_iteration;
    // unsigned em_iteration;
    // double floor;
    // GLLiMParameters<FullCovariance,FullCovariance> theta_expected;
    // GLLiMParameters<FullCovariance,FullCovariance> theta_trained = train(x, y, theta, kmeans_iteration, em_iteration, floor);
}

TEST_F(JGMMTest, getPosterior)
{
    // TODO
    // mat getPosterior();
}