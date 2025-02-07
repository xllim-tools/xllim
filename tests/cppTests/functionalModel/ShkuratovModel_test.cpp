#include <gtest/gtest.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "../../../src/functionalModel/ShkuratovModel.hpp"

namespace pt = boost::property_tree;

class ShkuratovModelTest : public testing::Test
{
protected:
    ShkuratovModelTest()
    {
        // Define Shkuratov parameters
        D = 50;
        scaling = {1.0, 1.5, 1.5, 1.5, 1.5};
        offset = {0, 0, 0.2, 0, 0};
        variant = "5p";

        // read geometries from dataset file
        pt::ptree root;
        pt::read_json("../tests/dataRef/Shkuratov5p_data_ref.json", root); // Load the json file in this ptree
        std::string variables[3] = {"inc", "eme", "phi"};
        geometries = mat(D, 3);
        unsigned i = 0;
        for (unsigned j = 0; j < 3; j++)
        {
            i = 0;
            for (pt::ptree::value_type &v : root.get_child(variables[j]))
            {
                geometries(i, j) = stod(v.second.data());
                i++;
            }
        }
        model = std::unique_ptr<ShkuratovModel>((new ShkuratovModel(geometries, variant, scaling, offset)));
    };

    std::unique_ptr<ShkuratovModel> model;
    unsigned D;
    mat geometries;
    vec scaling;
    vec offset;
    std::string variant;
};

TEST_F(ShkuratovModelTest, GetLDimension)
{
    ASSERT_EQ(model->getDimensionX(), 5);
}

TEST_F(ShkuratovModelTest, GetDDimension)
{
    ASSERT_EQ(model->getDimensionY(), D);
}

TEST_F(ShkuratovModelTest, ToPhysicOnes)
{
    vec x_true(5, fill::ones);
    x_true = x_true % scaling + offset;
    vec x(5, fill::ones);
    model->toPhysic(x);
    ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
}

TEST_F(ShkuratovModelTest, FromPhysicOnes)
{
    vec x_true(5, fill::ones);
    x_true = (x_true - offset) / scaling;
    vec x(5, fill::ones);
    model->fromPhysic(x);
    ASSERT_TRUE(approx_equal(x_true, x, "reldiff", 1e-8));
}

TEST_F(ShkuratovModelTest, FOnDataset)
{
    unsigned N = 10000;
    unsigned L = 5;
    unsigned n;
    unsigned d;
    pt::ptree root;
    pt::read_json("../tests/dataRef/Shkuratov5p_data_ref.json", root);

    // Read photometries
    mat photometries = mat(L, N);
    std::string variables[5] = {"an", "mu1", "nu", "m", "mu2"};
    for (unsigned l = 0; l < L; l++)
    {
        n = 0;
        for (pt::ptree::value_type &v : root.get_child(variables[l]))
        {
            photometries(l, n) = (stod(v.second.data()) - offset[l]) / scaling[l];
            n++;
        }
    }
    // Read expected results
    mat expected_results(D, N);
    n = 0;
    for (pt::ptree::value_type &v : root.get_child("y"))
    {
        d = 0;
        for (pt::ptree::value_type &w : v.second)
        {
            expected_results(d, n) = stod(w.second.data());
            d++;
        }
        n++;
    }
    // compute results from the models
    vec result(D);
    for (unsigned n = 0; n < N; n++)
    {
        model->F(photometries.col(n), result);
        ASSERT_TRUE(approx_equal(expected_results.col(n), result, "reldiff", 1e-8));
    }
}

TEST_F(ShkuratovModelTest, GetLDimension3p)
{
    model = std::unique_ptr<ShkuratovModel>((new ShkuratovModel(geometries, "3p", scaling, offset)));
    ASSERT_EQ(model->getDimensionX(), 3);
}
