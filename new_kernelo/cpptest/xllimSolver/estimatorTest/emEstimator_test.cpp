#include <gtest/gtest.h>
#include "../../../src/xllimSolver/emEstimator.hpp"

class EmEstimatorTest : public ::testing::Test
{
protected:
    // Constructor for initializing theta
    EmEstimatorTest() : theta(L, D, K) {}

    void SetUp() override
    {
        // Set A et B
        arma_rng::set_seed(10000);
        mat A(D, L, fill::randu);
        arma_rng::set_seed(11000);
        vec B(D, fill::randu);

        // Set sigma
        mat Sigma(N, D, fill::randn);
        Sigma *= sqrt(0.01);

        // Set C, Gamma et Pi
        arma_rng::set_seed(11100);
        vec Pi = normalise(vec(K, fill::randu), 1);

        arma_rng::set_seed(11110);
        mat C(L, K, fill::randu);

        cube Gamma(L, L, K);
        for (unsigned p = 0; p < K; p++)
        {
            arma_rng::set_seed_random();
            mat T_L = trimatl(mat(L, L, fill::randu));
            Gamma.slice(p) = T_L * T_L.t();
            Gamma.slice(p).diag() += 1;
        }

        // init gmm
        gmm_full model;
        model.set_params(C, Gamma, Pi.t());

        // sample X
        X = model.generate(N);

        // Compute Y
        Y = mat(A * X);
        for (unsigned n = 0; n < N; n++)
        {
            for (unsigned d = 0; d < D; d++)
            {
                Y(d, n) += B(d) + Sigma(n, d);
            }
        }

        mat sig(D, D, fill::zeros);
        sig.diag() += 0.01;

        theta.Pi = normalise(rowvec(K, fill::randu), 1);
        for (unsigned k = 0; k < K; k++)
        {
            theta.Sigma[k] = FullCovariance(sig);
            theta.Gamma[k] = FullCovariance(Gamma.slice(k));
            theta.A.slice(k) = A;
            theta.B.col(k) = B;
            theta.C.col(k) = C.col(k);
        }
    };

    const unsigned K = 2, N = 10000, L = 2, D = 5;
    GLLiMParameters<FullCovariance, FullCovariance> theta;
    EmEstimator<FullCovariance, FullCovariance> estimator;
    mat X, Y;
};

TEST_F(EmEstimatorTest, trainAndCheckParams)
{
    estimator.train(X, Y, theta, 5, 0.0, 1e-08);
    ASSERT_TRUE(theta.Pi.is_vec() && (theta.Pi.n_cols == K)) << "Pi dimensions must be of shape (" + std::to_string(K) + ")";
    ASSERT_TRUE(abs(accu(theta.Pi) - 1.0) < 1e-9) << "The sum of weights must be equal to 1";
    ASSERT_TRUE(arma::size(theta.A) == arma::SizeCube(D, L, K)) << "A dimensions must be of shape (" + std::to_string(D) + "," + std::to_string(L) + "," + std::to_string(K) + ")";
    ASSERT_TRUE(arma::size(theta.B) == arma::SizeMat(D, K)) << "B dimensions must be of shape (" + std::to_string(D) + "," + std::to_string(K) + ")";
    ASSERT_TRUE(arma::size(theta.C) == arma::SizeMat(L, K)) << "C dimensions must be of shape (" + std::to_string(L) + "," + std::to_string(K) + ")";
    ASSERT_TRUE(theta.Gamma.size() == K && arma::size(theta.Gamma[0].get_mat()) == arma::SizeMat(L, L)) << "Gamma dimensions must be of shape (" + std::to_string(L) + "," + std::to_string(L) + "," + std::to_string(K) + ")";
    ASSERT_TRUE(theta.Sigma.size() == K && arma::size(theta.Sigma[0].get_mat()) == arma::SizeMat(D, D)) << "Sigma dimensions must be of shape (" + std::to_string(D) + "," + std::to_string(D) + "," + std::to_string(K) + ")";
}

TEST_F(EmEstimatorTest, estimateZeroWeight)
{
    theta.Pi(1) = 0;
    estimator.train(X, Y, theta, 5, 0.0, 1e-08);
    ASSERT_EQ(theta.Pi(1), 0);
}

// TEST_F(EmEstimatorTest, estimateZeroDetMeans)
// {
//     for (unsigned k = 0; k < K; k++)
//     {
//         theta.Sigma[k] = FullCovariance(mat(D, D, fill::eye));
//     }
//     estimator.train(X, Y, theta, 5, 0.0, 1e-08);

//     ASSERT_EQ(accu(theta.Pi), 0);
// }

// TEST_F(EmEstimatorTest, estimateZeroDetCovariances)
// {
//     for (unsigned k = 0; k < K; k++)
//     {
//         theta.Gamma[k] = FullCovariance(mat(L, L, fill::eye));
//     }
//     estimator.train(X, Y, theta, 5, 0.0, 1e-08);

//     ASSERT_EQ(accu(theta.Pi), 0);
// }

// TEST_F(EmEstimatorTest, estimateZeroDetMeansZeroDetCovariances)
// {
//     for (unsigned k = 0; k < K; k++)
//     {
//         theta.Sigma[k] = FullCovariance(mat(D, D, fill::eye));
//         theta.Gamma[k] = FullCovariance(mat(L, L, fill::eye));
//     }
//     estimator.train(X, Y, theta, 5, 0.0, 1e-08);

//     ASSERT_EQ(accu(theta.Pi), 0);
// }

// TEST_F(EmEstimatorTest, update_x)
// {
//     // TODO private methods
// }
