#include <gtest/gtest.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "../../src/functionalModel/ExternalPythonModel.hpp"


namespace pt = boost::property_tree;


namespace Functional{
    
    class ExternalShkuratovModel5pTest : public testing::Test{
    protected:

        ExternalShkuratovModel5pTest() {
            // Python interpreter must be initialize here in order to tell CTest to run the Python interpreter in its test environment
            Py_Initialize();
        }
        void SetUp() override {

            // Define Shkuratov parameters as described in ShkuratovModel5pPython.py
            D = 50;
            scaling = {1.0,1.5,1.5,1.5,1.5};
            offset = {0,0,0.2,0,0};
            model = std::unique_ptr<ExternalPythonModel>((new ExternalPythonModel("ShkuratovModel5p", "ShkuratovModel5pPython", "../../pytest/models/")));
            std::cout << "extern" <<std::endl;
        };

        std::unique_ptr<ExternalPythonModel> model;
        unsigned D;
        vec scaling;
        vec offset;
        std::string variant;
    };

    TEST_F(ExternalShkuratovModel5pTest, GetLDimension){
        ASSERT_EQ(model->get_L_dimension(), 5);
    }
    
    TEST_F(ExternalShkuratovModel5pTest, GetDDimension){
        ASSERT_EQ(model->get_D_dimension(), D);
    }

    TEST_F(ExternalShkuratovModel5pTest, ToPhysicOnes){
        vec x_true(5, fill::ones);
        x_true = x_true % scaling + offset;
        vec x(5, fill::ones);
        model->to_physic(x);
        ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
    }
    
    TEST_F(ExternalShkuratovModel5pTest, FromPhysicOnes){
        vec x_true(5, fill::ones);
        x_true = (x_true - offset) / scaling;
        vec x(5, fill::ones);
        model->from_physic(x);
        ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
    }

    TEST_F(ExternalShkuratovModel5pTest, FOnDataset){
        unsigned N = 10000;
        unsigned L = 5;
        unsigned n;
        unsigned d;
        pt::ptree root;
        pt::read_json("../cpptest/functionalModel/dataRef/Shkuratov5p_data_ref.json", root);
        
        // Read photometries
        mat photometries = mat(L,N);
        std::string variables[5] = {"an", "mu1", "nu", "m", "mu2"};
        for(unsigned l=0; l < L; l++){
            n=0;
            for (pt::ptree::value_type& v : root.get_child(variables[l]))
            {
                photometries(l,n) = (stod(v.second.data()) - offset[l]) / scaling[l];
                n++;
            }
        }
        // Read expected results
        mat expected_results(D,N);
        n=0;
        for (pt::ptree::value_type& v : root.get_child("y"))
        {
            d=0;
            for (pt::ptree::value_type& w : v.second){
                expected_results(d,n) = stod(w.second.data());
                d++;
            }
            n++;
        }
        // compute results from the models
        vec result(D);
        for(unsigned n=0; n<N; n++){
            model->F(photometries.col(n), result);
            ASSERT_TRUE(approx_equal(expected_results.col(n), result, "reldiff", 1e-8));
        }
    }

}



// int main(int argc, char **argv){
//     testing::InitGoogleTest(&argc, argv);
//     // testing::GTEST_FLAG(filter) = "*HapkeFunctionalTest*";
//     return RUN_ALL_TESTS();
// }