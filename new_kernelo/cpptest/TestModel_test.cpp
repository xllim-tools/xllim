#include <gtest/gtest.h>
#include "../src/TestModel.hpp"

using namespace std;

namespace Functional{
    
    class TestModelTest : public testing::Test{
    protected:

        void SetUp() override {
            model = unique_ptr<TestModel>((new TestModel()));
        }

        unique_ptr<TestModel> model;
    };

    TEST_F(TestModelTest, GetLDimension){
        ASSERT_EQ(model->get_L_dimension(), 4);
    }

    TEST_F(TestModelTest, GetDDimension){
        ASSERT_EQ(model->get_D_dimension(), 9);
    }

    TEST_F(TestModelTest, ToPhysicOnes){
        vec x_true(4, fill::ones);
        x_true *= 2;
        vec x(4, fill::ones);
        model->to_physic(x);
        ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
    }
    
    TEST_F(TestModelTest, FromPhysicOnes){
        vec x_true(4, fill::ones);
        x_true /= 2;
        vec x(4, fill::ones);
        model->from_physic(x);
        ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
    }

    TEST_F(TestModelTest, FOnZerosX){
        vec y_true = {4+2*datum::e, 0.5, datum::e, 3, 0.2, -0.5, -0.2-datum::e, 2*datum::e-1, -0.7};
        y_true *= 0.5;
        vec x(4, fill::zeros);
        vec y(9, fill::ones);
        model->F(x, y);
        ASSERT_TRUE(approx_equal(y_true, y, "reldiff", 1e-8));
    }
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}