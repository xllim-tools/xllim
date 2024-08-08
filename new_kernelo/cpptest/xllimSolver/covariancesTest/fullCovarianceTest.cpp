#include <gtest/gtest.h>
#include "../../../src/xllimSolver/covariances/covariance.hpp"

class FullCovarianceTest : public ::testing::Test
{
protected:
    FullCovarianceTest()
    {
        A_arma = mat({{1, 2, 3, 4},
                      {2, 6, 7, 8},
                      {3, 7, 11, 12},
                      {4, 8, 12, 16}});

        B_arma = mat({{4, 1, 1, 1},
                      {1, 5, 1, 1},
                      {1, 1, 6, 1},
                      {1, 1, 1, 7}});
    }
    mat A_arma;
    mat B_arma;
};

TEST_F(FullCovarianceTest, Constructors)
{
    FullCovariance A_full(A_arma);
    ASSERT_EQ(accu(A_full.get_mat() != A_arma), 0);

    FullCovariance B_full(5);
    ASSERT_EQ(accu(B_full.get_mat() != mat(5, 5, fill::eye)), 0);
}

TEST_F(FullCovarianceTest, EqualOperator)
{
    FullCovariance A_full;
    FullCovariance B_full(B_arma);

    A_full = A_arma;
    ASSERT_EQ(accu(A_arma != A_full.get_mat()), 0);

    A_full = B_full;
    ASSERT_EQ(accu(B_arma != A_full.get_mat()), 0);
}

TEST_F(FullCovarianceTest, IncrementOperator)
{
    FullCovariance A_full(A_arma);
    A_full += mat(4, 4, fill::ones);
    mat expected_result = A_arma + mat(4, 4, fill::ones);
    ASSERT_EQ(accu(A_full.get_mat() != expected_result), 0);

    A_full += 1;
    expected_result += 1;
    ASSERT_EQ(accu(A_full.get_mat() != expected_result), 0);
}

TEST_F(FullCovarianceTest, AdditionOperator)
{
    mat a_arma(4, 4, fill::randu);
    FullCovariance A_full(A_arma);
    mat result_lefteft = A_full + a_arma;
    mat result_rightight = a_arma + A_full;
    mat expected_result = a_arma + A_arma;

    ASSERT_EQ(accu(result_lefteft != expected_result), 0);
    ASSERT_EQ(accu(result_rightight != expected_result), 0);
}

TEST_F(FullCovarianceTest, SubtractionOperator)
{
    mat a_arma(4, 4, fill::ones);
    FullCovariance A_full(A_arma);
    mat result_left = A_full - a_arma;
    mat result_right = a_arma - A_full;
    mat expected_result_left = mat({{0, 1, 2, 3},
                                    {1, 5, 6, 7},
                                    {2, 6, 10, 11},
                                    {3, 7, 11, 15}});
    mat expected_result_right = mat({{0, -1, -2, -3},
                                     {-1, -5, -6, -7},
                                     {-2, -6, -10, -11},
                                     {-3, -7, -11, -15}});

    ASSERT_EQ(accu(result_left != expected_result_left), 0);
    ASSERT_EQ(accu(result_right != expected_result_right), 0);
}

TEST_F(FullCovarianceTest, ProductOperator)
{
    FullCovariance B_full(A_arma);
    mat a_arma(4, 6, fill::ones);
    mat b_arma(6, 4, fill::ones);

    mat result_left = B_full * a_arma;
    mat result_right = b_arma * B_full;
    vec result_vec = B_full * vec(4, fill::ones);
    vec expected_result_vec({10., 23., 33., 40.});
    rowvec result_rowvec = rowvec(4, fill::ones) * B_full;
    rowvec expected_result_rowvec({10., 23., 33., 40.});

    ASSERT_EQ(accu(result_left != A_arma * a_arma), 0);
    ASSERT_EQ(accu(result_right != b_arma * A_arma), 0);
    ASSERT_EQ(accu(result_vec != expected_result_vec), 0);
    ASSERT_EQ(accu(result_rowvec != expected_result_rowvec), 0);
}

TEST_F(FullCovarianceTest, logDet)
{
    FullCovariance B_full(B_arma);
    double result = B_full.log_det();
    double expected_result = 6.553933404025811;
    ASSERT_DOUBLE_EQ(result, expected_result);
}

TEST_F(FullCovarianceTest, inv)
{
    FullCovariance B_full(B_arma);
    FullCovariance result = B_full.inv();
    mat expected_result = mat({{0.27635328, -0.04273504, -0.03418803, -0.02849003},
                               {-0.04273504, 0.21794872, -0.02564103, -0.02136752},
                               {-0.03418803, -0.02564103, 0.17948718, -0.01709402},
                               {-0.02849003, -0.02136752, -0.01709402, 0.15242165}});
    ASSERT_TRUE(approx_equal(result.get_mat(), expected_result, "absdiff", 1e-3));
}

TEST_F(FullCovarianceTest, rankOneUpdate)
{
    // TODO
    // FullCovariance A_full(B_arma);
    // A_full.rankOneUpdate(vec(4, fill::ones), 0.1);
    // mat result = A_full.get_mat();
    // mat expected_result = B_arma + mat(4, 4, fill::ones) * 0.1;

    // ASSERT_EQ(accu(expected_result != result), 0);
}

TEST_F(FullCovarianceTest, print)
{
    FullCovariance A_full(A_arma);
    A_full.print("FullCovarianceTest/print A_full = ");
}
