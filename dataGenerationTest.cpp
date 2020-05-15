//
// Created by reverse-proxy on 6‏/1‏/2020.
//

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <iostream>
#include <chrono>
#include "src/functionalModel/FunctionalModel.h"
#include "src/functionalModel/FunctionnalModelFactory.h"
#include "src/functionalModel/HapkeModel/HapkeVersions/Hapke02Model.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/FourParamsModel.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.h"
#include "src/functionalModel/HapkeModel/HapkeAdapters/SixParamsModel.h"
#include "src/dataGeneration//LatinCubeGenerator.h"
#include "src/dataGeneration/creators.h"

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


    std::shared_ptr<FunctionalModel> myModel (new Hapke02Model(geometries, 50, 3, std::shared_ptr<HapkeAdapter>(new SixParamsModel()), 30.0));
    std::shared_ptr<DataGeneration::StatModel> statModel = DataGeneration::DependentGaussianStatModelConfig("sobol", myModel,20, 123456789).create();

    auto start = chrono::high_resolution_clock::now();
    statModel->gen_data(10000);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << duration.count() << endl;

    delete[] x;
    delete[] y;


}

