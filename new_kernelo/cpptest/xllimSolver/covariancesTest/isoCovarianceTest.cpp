#include <gtest/gtest.h>
#include "../../../src/xllimSolver/covariances/covariance.hpp"

class IsoCovarianceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        A_mat = mat({{25, 0, 0, 0},
                     {0, 25, 0, 0},
                     {0, 0, 25, 0},
                     {0, 0, 0, 25}});

        A_vec = vec({25, 25, 25, 25});

        A_scalar = 25;
    }
    mat A_mat;
    vec A_vec;
    double A_scalar;
};

TEST_F(IsoCovarianceTest, Constructor)
{
    IsoCovariance A_iso(A_mat);
    ASSERT_EQ(accu(A_iso.get_mat() != A_mat), 0);

    IsoCovariance B_iso(5);
    ASSERT_EQ(accu(B_iso.get_mat() != mat(5, 5, fill::eye)), 0);

    IsoCovariance C_iso(A_scalar, 4);
    ASSERT_EQ(accu(C_iso.get_mat() != A_mat), 0);
}

TEST_F(IsoCovarianceTest, EqualOperator)
{
    IsoCovariance A_iso;
    IsoCovariance B_iso(A_scalar, 4);
    
    A_iso = B_iso;
    ASSERT_EQ(accu(A_iso.get_mat() != A_mat), 0);

    A_iso = A_mat;
    ASSERT_EQ(accu(A_iso.get_mat() != A_mat), 0);

    B_iso = 5;
    ASSERT_EQ(accu(B_iso.get_vec() != vec({5, 5, 5, 5})), 0);
}

TEST_F(IsoCovarianceTest, IncrementOperator)
{
    IsoCovariance A_iso(A_scalar, 4);
    A_iso += mat(4, 4, fill::ones);
    mat expected_result = A_mat;
    expected_result.diag() += 1;
    ASSERT_EQ(accu(A_iso.get_mat() != expected_result), 0);

    A_iso += 8.6;
    expected_result.diag() += 8.6;
    ASSERT_EQ(accu(A_iso.get_mat() != expected_result), 0);
}

TEST_F(IsoCovarianceTest, AdditionOperator)
{
    mat a_arma(4, 4, fill::ones);
    IsoCovariance A_iso(A_mat);
    mat result_left = A_iso + a_arma;
    mat result_right = a_arma + A_iso;
    mat expected_result = a_arma + A_mat;

    ASSERT_EQ(accu(result_left != result_right), 0);
    ASSERT_EQ(accu(result_left != expected_result), 0);
}

TEST_F(IsoCovarianceTest, SubtractionOperator)
{
    mat a_arma(4, 4, fill::ones);
    IsoCovariance A_iso(A_mat);
    mat result_left = A_iso - a_arma;
    mat result_right = a_arma - A_iso;
    mat expected_result_left = mat({{24, -1, -1, -1},
                                    {-1, 24, -1, -1},
                                    {-1, -1, 24, -1},
                                    {-1, -1, -1, 24}});
    mat expected_result_right = mat({{-24, 1, 1, 1},
                                     {1, -24, 1, 1},
                                     {1, 1, -24, 1},
                                     {1, 1, 1, -24}});

    ASSERT_EQ(accu(result_left != expected_result_left), 0);
    ASSERT_EQ(accu(result_right != expected_result_right), 0);
}

TEST_F(IsoCovarianceTest, ProductOperator)
{
    IsoCovariance A_iso(A_mat);
    mat a_arma(4, 6, fill::ones);
    mat b_arma(6, 4, fill::ones);

    mat result_left = A_iso * a_arma;
    mat result_right = b_arma * A_iso;
    vec result_vec = A_iso * vec(4, fill::ones);
    vec expected_result_vec({25, 25, 25, 25});
    rowvec result_rowvec = rowvec(4, fill::ones) * A_iso;
    rowvec expected_result_rowvec({25, 25, 25, 25});

    ASSERT_EQ(accu(result_left != A_mat * a_arma), 0);
    ASSERT_EQ(accu(result_right != b_arma * A_mat), 0);
    ASSERT_EQ(accu(result_vec != expected_result_vec), 0);
    ASSERT_EQ(accu(result_rowvec != expected_result_rowvec), 0);
}

TEST_F(IsoCovarianceTest, logDet)
{
    IsoCovariance A_iso(A_scalar, 4);
    double result = A_iso.log_det();
    double expected_result = 4.605170185988091;
    ASSERT_DOUBLE_EQ(result, expected_result);
}

TEST_F(IsoCovarianceTest, inv)
{
    IsoCovariance A_iso(A_scalar, 4);
    IsoCovariance result = A_iso.inv();
    vec expected_result = vec({0.04, 0.04, 0.04, 0.04});
    ASSERT_EQ(accu(result.get_vec() != expected_result), 0);
}

TEST_F(IsoCovarianceTest, rankOneUpdate)
{
    // IsoCovariance A_iso(A_mat);
    // A_iso.rankOneUpdate(vec(4, fill::ones), 0.1);
    // mat result = A_iso.get_mat();
    // mat expected_result = mat({{25.1, 0, 0, 0},
    //                            {0, 25.1, 0, 0},
    //                            {0, 0, 25.1, 0},
    //                            {0, 0, 0, 25.1}});

    // ASSERT_EQ(accu(expected_result != result), 0);
}

TEST_F(IsoCovarianceTest, print)
{
    IsoCovariance A_iso(A_scalar, 4);
    A_iso.print("IsoCovarianceTest/print A_iso = ");
}
