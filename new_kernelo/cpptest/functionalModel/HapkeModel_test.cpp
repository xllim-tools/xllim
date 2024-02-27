#include <gtest/gtest.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "../src/HapkeModel.hpp"


namespace pt = boost::property_tree;


namespace Functional{

// ******************************************************************** //
//                      Hapke 6 Parameters Tests
// ******************************************************************** //
    
    class HapkeModel6ParamsTest : public testing::Test{
    protected:
        void SetUp() override {

            // Define Hapke parameters
            L = 6;
            D = 70;
            variant = "2002";
            adapter = "six";
            theta_bar_scaling = 30.0;
            b0 = 1;
            h = 0;

            //Read geometries for 6 parameters model
            pt::ptree root;
            pt::read_json("../cpptest/Hapke6p_geom70_data_ref.json", root);  // Load the json file in this ptree
            geometries = mat(D, 3);
            unsigned i = 0;
            unsigned j = 0;
            for (pt::ptree::value_type& v : root.get_child("data_ref").get_child("geometries"))
            {
                j = 0;
                for (auto it: v.second.get_child("")) {
                    geometries(i,j) = stod(it.second.data());
                    j++;
                }
                i++;
            }
            model = std::unique_ptr<HapkeModel>(new HapkeModel(geometries, variant, adapter, theta_bar_scaling, b0, h));
        };

        std::unique_ptr<HapkeModel> model;
        unsigned D;
        unsigned L;
        mat geometries;
        std::string variant;
        std::string adapter;
        double theta_bar_scaling;
        double b0;
        double h;
    };

    TEST_F(HapkeModel6ParamsTest, GetLDimension){
        ASSERT_EQ(model->get_L_dimension(), L);
    }
    
    TEST_F(HapkeModel6ParamsTest, GetDDimension){
        ASSERT_EQ(model->get_D_dimension(), D);
    }

    TEST_F(HapkeModel6ParamsTest, ToPhysicOnesTenths){
        vec x_true(L, fill::ones);
        x_true *= 0.1;
        x_true(0) = 0.19; // x(0) corresponds to OMEGA : x(OMEGA) = 1 - pow(1 - x(OMEGA), 2);
        x_true(1) *= theta_bar_scaling; // x(1) corresponds to THETA_BAR : x(THETA_BAR) *= theta_bar_scaling;
        vec x(L, fill::ones);
        x *= 0.1;
        model->to_physic(x);
        ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
    }
    
    TEST_F(HapkeModel6ParamsTest, FromPhysicOneTenths){
        vec x_true(L, fill::ones);
        x_true *= 0.1;
        x_true(0) = 1 - sqrt(1 - x_true(0)); // x(0) corresponds to OMEGA : x(OMEGA) = 1 - sqrt(1 - x(OMEGA));
        x_true(1) /= theta_bar_scaling; // x(1) corresponds to THETA_BAR : x(THETA_BAR) /= theta_bar_scaling;
        vec x(L, fill::ones);
        x *= 0.1;
        model->from_physic(x);
        ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
    }

    TEST_F(HapkeModel6ParamsTest, FOnDataset){
        unsigned N = 20000;
        unsigned n, d, l;
        pt::ptree root;
        pt::read_json("../cpptest/Hapke6p_geom70_data_ref.json", root);
        
        // Read photometries
        mat photometries(L,N);
        n=0;
        for (pt::ptree::value_type& v : root.get_child("data_ref").get_child("synthetic_dataset").get_child("X")) {
            l=0;
            for (auto it: v.second.get_child("")) {
                photometries(l,n) = stod(it.second.data());
                l++;
            }
            n++;
        }
        // Read expected results
        mat expected_results(D,N);
        n=0;
        for (pt::ptree::value_type& v : root.get_child("data_ref").get_child("synthetic_dataset").get_child("Y"))
        {
            d=0;
            for (auto it: v.second.get_child("")) {
                expected_results(d,n) = stod(it.second.data());
                d++;
            }
            n++;
        }
        // compute results from the models
        vec result(D);
        for(unsigned n=0; n<N; n++){
            model->F(photometries.col(n), result);
            ASSERT_TRUE(approx_equal(expected_results.col(n), result, "reldiff", 1e-7)); // it fails at 1e-8
        }
    }


// ******************************************************************** //
//                      Hapke 4 Parameters Tests
// ******************************************************************** //

    class HapkeModel4ParamsTest : public testing::Test{
    protected:
        void SetUp() override {

            // Define Hapke parameters
            L = 4;
            D = 70;
            variant = "2002";
            adapter = "four";
            theta_bar_scaling = 30.0;
            b0 = 1;
            h = 0;

            //Read geometries for 6 parameters model
            pt::ptree root;
            pt::read_json("../cpptest/Hapke4p_geom70_data_ref.json", root);  // Load the json file in this ptree
            geometries = mat(D, 3);
            unsigned i = 0;
            unsigned j = 0;
            for (pt::ptree::value_type& v : root.get_child("data_ref").get_child("geometries"))
            {
                j = 0;
                for (auto it: v.second.get_child("")) {
                    geometries(i,j) = stod(it.second.data());
                    j++;
                }
                i++;
            }
            model = std::unique_ptr<HapkeModel>(new HapkeModel(geometries, variant, adapter, theta_bar_scaling, b0, h));
        };

        std::unique_ptr<HapkeModel> model;
        unsigned D;
        unsigned L;
        mat geometries;
        std::string variant;
        std::string adapter;
        double theta_bar_scaling;
        double b0;
        double h;
    };

    TEST_F(HapkeModel4ParamsTest, GetLDimension){
        ASSERT_EQ(model->get_L_dimension(), L);
    }
    
    TEST_F(HapkeModel4ParamsTest, GetDDimension){
        ASSERT_EQ(model->get_D_dimension(), D);
    }

    TEST_F(HapkeModel4ParamsTest, ToPhysicOnesTenths){
        vec x_true(L, fill::ones);
        x_true *= 0.1;
        x_true(0) = 0.19; // x(0) corresponds to OMEGA : x(OMEGA) = 1 - pow(1 - x(OMEGA), 2);
        x_true(1) *= theta_bar_scaling; // x(1) corresponds to THETA_BAR : x(THETA_BAR) *= theta_bar_scaling;
        vec x(L, fill::ones);
        x *= 0.1;
        model->to_physic(x);
        ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
    }
    
    TEST_F(HapkeModel4ParamsTest, FromPhysicOneTenths){
        vec x_true(L, fill::ones);
        x_true *= 0.1;
        x_true(0) = 1 - sqrt(1 - x_true(0)); // x(0) corresponds to OMEGA : x(OMEGA) = 1 - sqrt(1 - x(OMEGA));
        x_true(1) /= theta_bar_scaling; // x(1) corresponds to THETA_BAR : x(THETA_BAR) /= theta_bar_scaling;
        vec x(L, fill::ones);
        x *= 0.1;
        model->from_physic(x);
        ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
    }

    TEST_F(HapkeModel4ParamsTest, FOnDataset){
        unsigned N = 20000;
        unsigned n, d, l;
        pt::ptree root;
        pt::read_json("../cpptest/Hapke4p_geom70_data_ref.json", root);
        
        // Read photometries
        mat photometries(L,N);
        n=0;
        for (pt::ptree::value_type& v : root.get_child("data_ref").get_child("synthetic_dataset").get_child("X")) {
            l=0;
            for (auto it: v.second.get_child("")) {
                photometries(l,n) = stod(it.second.data());
                l++;
            }
            n++;
        }
        // Read expected results
        mat expected_results(D,N);
        n=0;
        for (pt::ptree::value_type& v : root.get_child("data_ref").get_child("synthetic_dataset").get_child("Y"))
        {
            d=0;
            for (auto it: v.second.get_child("")) {
                expected_results(d,n) = stod(it.second.data());
                d++;
            }
            n++;
        }
        // compute results from the models
        vec result(D);
        for(unsigned n=0; n<N; n++){
            model->F(photometries.col(n), result);
            ASSERT_TRUE(approx_equal(expected_results.col(n), result, "reldiff", 1e-7)); // it fails at 1e-8
        }
    }


}

