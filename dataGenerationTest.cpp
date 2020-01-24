//
// Created by reverse-proxy on 6‏/1‏/2020.
//

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <iostream>
#include <chrono>
#include "src/physicalModel/FunctionalModel.h"
#include "src/physicalModel/FunctionnalModelFactory.h"
#include "src/physicalModel/Hapke02Model.h"
#include "src/physicalModel/FourParamsModel.h"
#include "src/physicalModel/ThreeParamsModel.h"
#include "src/physicalModel/SixParamsModel.h"
#include "src/dataGeneration//LatinCubeGenerator.h"

#include <iostream>
#include <cstring>
#include <exception>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "src/dataGeneration/GaussianStatModel.h"
#include "src/dataGeneration/DependentGaussianStatModel.h"

using namespace std;
namespace pt = boost::property_tree;

using namespace boost::random;
typedef sobol_engine< boost::uint_least64_t, 64u, default_sobol_table > Sobol;

int main(){
    static const std::size_t dimension = 6;

    auto *geometries = new double[50*3];
    unsigned i = 0;


    pt::ptree root;
    pt::read_json("../test_hapke.json", root);  // Load the json file in this ptree
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("eme"))
    {
        geometries[i*3+0] = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("inc"))
    {
        geometries[i*3+1] = stod(v.second.data());

        i++;
    }
    i = 0;



    for (pt::ptree::value_type& v : root.get_child("phi"))
    {
        geometries[i*3+2] = stod(v.second.data());
        i+=1;
    }

    double cov[50];
    std::fill_n(cov,50,1.0/400);
    auto *x = new double[6*10000];
    auto *y = new double[50*10000];


    std::shared_ptr<FunctionalModel> myModel (new Hapke02Model(geometries, 50, 3, std::shared_ptr<HapkeAdapter>(new ThreeParamsModel(0.0,0.1)), 30.0));
    DataGeneration::DependentGaussianStatModel statModel = DataGeneration::DependentGaussianStatModel("sobol", 20, 123456789);

    std::get<1>(statModel.gen_data(myModel, 1)).print();


    delete[] x;
    delete[] y;


}

