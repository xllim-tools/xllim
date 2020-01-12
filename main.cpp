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



    std::shared_ptr<FunctionnalModel> myModel = FunctionnalModelFactory::getModel("hapke02", geometries,50,3);

    auto *x = new double[6];
    for(unsigned j=0; j<6; j++){
        x[j] = photometries(0,j);
    }

    auto *y = new double[50];

    myModel->F(x, 6, y, 50);

    cout << y[0] << endl;

    cout << myModel->get_D_dimension() <<'\n';
    cout << myModel->get_L_dimension() <<'\n';

    delete [] x;
    delete [] y;
    delete [] geometries;




    return 0;
}