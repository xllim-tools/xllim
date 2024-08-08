#include <gtest/gtest.h>
#include "../../../src/xllimSolver/covariances/covariance.hpp"

class DiagCovarianceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        A_mat = mat({{1, 0, 0, 0},
                     {0, 2, 0, 0},
                     {0, 0, 3, 0},
                     {0, 0, 0, 4}});

        B_vec = vec({1., 0.5, 10., 0.02});
    }
    mat A_mat;
    vec B_vec;
};

TEST_F(DiagCovarianceTest, Constructor)
{
    DiagCovariance A_diag(B_vec);
    ASSERT_EQ(accu(A_diag.get_vec() != B_vec), 0);

    DiagCovariance B_diag(A_mat);
    ASSERT_EQ(accu(B_diag.get_mat() != A_mat), 0);

    DiagCovariance C_diag(5);
    ASSERT_EQ(accu(C_diag.get_mat() != mat(5, 5, fill::eye)), 0);
}

TEST_F(DiagCovarianceTest, EqualOperator)
{
    DiagCovariance A_diag;
    DiagCovariance B_diag(B_vec);

    A_diag = B_diag;
    ASSERT_EQ(accu(A_diag.get_vec() != B_diag.get_vec()), 0);

    A_diag = A_mat;
    ASSERT_EQ(accu(A_diag.get_mat() != A_mat), 0);

    A_diag = B_vec;
    ASSERT_EQ(accu(A_diag.get_vec() != B_vec), 0);
}

TEST_F(DiagCovarianceTest, IncrementOperator)
{
    DiagCovariance A_diag(A_mat);
    A_diag += mat(4, 4, fill::ones);
    mat expected_result = A_mat;
    expected_result.diag() += 1;
    ASSERT_EQ(accu(A_diag.get_mat() != expected_result), 0);

    A_diag += 8.6;
    expected_result.diag() += 8.6;
    ASSERT_EQ(accu(A_diag.get_mat() != expected_result), 0);
}

TEST_F(DiagCovarianceTest, AdditionOperator)
{
    mat a_arma(4, 4, fill::ones);
    DiagCovariance A_diag(A_mat);
    mat result_left = A_diag + a_arma;
    mat result_right = a_arma + A_diag;
    mat expected_result = a_arma + A_mat;

    ASSERT_EQ(accu(result_left != result_right), 0);
    ASSERT_EQ(accu(result_left != expected_result), 0);
}

TEST_F(DiagCovarianceTest, SubtractionOperator)
{
    mat a_arma(4, 4, fill::ones);
    DiagCovariance A_diag(A_mat);
    mat result_left = A_diag - a_arma;
    mat result_right = a_arma - A_diag;
    mat expected_result_left = mat({{0, -1, -1, -1},
                                    {-1, 1, -1, -1},
                                    {-1, -1, 2, -1},
                                    {-1, -1, -1, 3}});
    mat expected_result_right = mat({{0, 1, 1, 1},
                                     {1, -1, 1, 1},
                                     {1, 1, -2, 1},
                                     {1, 1, 1, -3}});

    ASSERT_EQ(accu(result_left != expected_result_left), 0);
    ASSERT_EQ(accu(result_right != expected_result_right), 0);
}

TEST_F(DiagCovarianceTest, ProductOperator)
{
    DiagCovariance A_diag(A_mat);
    mat a_arma(4, 6, fill::ones);
    mat b_arma(6, 4, fill::ones);

    mat result_left = A_diag * a_arma;
    mat result_right = b_arma * A_diag;
    vec result_vec = A_diag * vec({10, -2, 0.3, 4});
    vec expected_result_vec({10., -4., 0.9, 16.});
    rowvec result_rowvec = rowvec({10, -2, 0.3, 4}) * A_diag;
    rowvec expected_result_rowvec({10., -4., 0.9, 16.});

    ASSERT_EQ(accu(result_left != A_mat * a_arma), 0);
    ASSERT_EQ(accu(result_right != b_arma * A_mat), 0);
    // ASSERT_EQ(accu(result_vec != expected_result_vec), 0);
    ASSERT_TRUE(approx_equal(result_vec, expected_result_vec, "absdiff", 1e-8));
    // ASSERT_EQ(accu(result_rowvec != expected_result_rowvec), 0);
    ASSERT_TRUE(approx_equal(result_rowvec, expected_result_rowvec, "absdiff", 1e-8));
}

TEST_F(DiagCovarianceTest, logDet)
{
    DiagCovariance B_diag(B_vec);
    double result = B_diag.log_det();
    double expected_result = -2.3025850929940455;
    ASSERT_DOUBLE_EQ(result, expected_result);
}

TEST_F(DiagCovarianceTest, inv)
{
    DiagCovariance B_diag(B_vec);
    DiagCovariance result = B_diag.inv();
    vec expected_result = vec({1., 2., 0.1, 50.});
    ASSERT_EQ(accu(result.get_vec() != expected_result), 0);
}

TEST_F(DiagCovarianceTest, rankOneUpdate)
{
    // TODO
    // DiagCovariance A_diag(B_vec);
    // A_diag.rankOneUpdate(vec(4, fill::ones), 0.1);
    // mat result = A_diag.get_mat();
    // mat expected_result = mat({{1.1, 0, 0, 0},
    //                            {0, 6.1, 0, 0},
    //                            {0, 0, 11.1, 0},
    //                            {0, 0, 0, 16.1}});

    // ASSERT_EQ(accu(expected_result != result), 0);
}

TEST_F(DiagCovarianceTest, print)
{
    DiagCovariance B_diag(B_vec);
    B_diag.print("DiagCovarianceTest/print B_diag = ");
}
