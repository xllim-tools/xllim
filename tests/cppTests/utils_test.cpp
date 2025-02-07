#include <gtest/gtest.h>
#include "../../src/utils/utils.hpp"

using namespace utils;

class UtilsTest : public testing::Test
{
protected:
    UtilsTest()
    {
    }
};

TEST_F(UtilsTest, LogSumExpOnVec)
{
    // const vec &x
    const vec x({-33, -12.444, -166, -13.7, -22.5});
    double result = logSumExp(x);
    double true_result = -12.193370594714498;
    ASSERT_NEAR(result, true_result, 1e-5); // abs_error
}

TEST_F(UtilsTest, LogSumExpOnMat)
{
    const mat x = {{-33, -12.444, -166, -13.7, -22.5},
                   {-33, -12.444, -166, -13.7, -22.5},
                   {-33, -12.444, -166, -13.7, -22.5}};
    vec true_result = {-12.193370594714498, -12.193370594714498, -12.193370594714498};
    vec result = logSumExp(x, 1);
    ASSERT_TRUE(approx_equal(result, true_result, "reldiff", 1e-8));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}