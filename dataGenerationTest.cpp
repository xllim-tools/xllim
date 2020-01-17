//
// Created by reverse-proxy on 6‏/1‏/2020.
//

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <iostream>
#include <chrono>
#include "src/physicalModel/FunctionnalModel.h"
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

    // Create a generator
    typedef boost::variate_generator<Sobol, boost::uniform_01<double>> quasi_random_gen_t;

    // Initialize the engine to draw randomness out of thin air
    Sobol engine(dimension);

    std::vector<double> sample(dimension);

    // At this point you can use std::generate, generate member f-n, etc.
    //engine.generate(sample.begin(), sample.end());


    //-------------------------------------------------------
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(seed & 0xffffffff), uint32_t(seed>>32)};

    // mt19937 is a standard mersenne_twister_engine
    std::mt19937_64 generator;
    generator.seed(ss);

    std::uniform_real_distribution<double> unif(0, 1);
    // ready to generate random numbers
    for (int i = 0; i < 20; i++)
    {
        for(int j=0; j<6; j++){
            //double currentRandomNumber = unif(generator);
            //std::cout << currentRandomNumber << " ";
        }
        //std::cout << std::endl;
    }

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
    auto *x = new double[6*10];
    auto *y = new double[50*10];
    std::shared_ptr<FunctionnalModel> myModel (new Hapke02Model(geometries, 50, 3, std::shared_ptr<HapkeAdapter>(new SixParamsModel())));
    DataGeneration::DependentGaussianStatModel statModel = DataGeneration::DependentGaussianStatModel("latin_cube",20.0);
    statModel.gen_data(myModel,10,x,y);

    delete[] x;
    delete[] y;


}

