//
// Created by reverse-proxy on 18‏/12‏/2019.
//



#include <armadillo>
#include "src/physicalModel/FunctionnalModel.h"
#include "src/physicalModel/FunctionnalModelFactory.h"
#include <utility>

#include <iostream>
#include <cstring>
#include <exception>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <fstream>


using namespace arma;

using namespace std;
namespace pt = boost::property_tree;


int main(){


    std::vector<std::vector<double>> geometries(50,std::vector<double>(3));
    unsigned i = 0;


    pt::ptree root;
    pt::read_json("../test_hapke.json", root);  // Load the json file in this ptree
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("eme"))
    {
        geometries[i][0] = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("inc"))
    {
        geometries[i][1] = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("phi"))
    {
        geometries[i][2] = stod(v.second.data());
        i+=1;
    }

    mat photometries = mat(10000,6);

    i = 0;
    for (pt::ptree::value_type& v : root.get_child("omega"))
    {
        photometries(i,0) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("b"))
    {
        photometries(i,2) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("c"))
    {
        photometries(i,3) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("hh"))
    {
        photometries(i,5) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("b0"))
    {
        photometries(i,4) = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("theta0"))
    {
        photometries(i,1) = stod(v.second.data());
        i++;
    }

    
    std::shared_ptr<FunctionnalModel> myModel = FunctionnalModelFactory::getModel("hapke02", geometries);

    std::vector<std::vector<double>> x(photometries.n_rows);
    for(unsigned j=0; j<photometries.n_rows; j++){
        x[j] = conv_to< std::vector<double> >::from(photometries.row(j));
    }

    cout << myModel->F(x[658])[3]<<'\n';

    cout << myModel->get_D_dimension() <<'\n';
    cout << myModel->get_L_dimension() <<'\n';



    return 0;
}