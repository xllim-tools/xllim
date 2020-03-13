//
// Created by reverse-proxy on 18‏/12‏/2019.
//



#include <armadillo>
#include "src/physicalModel/FunctionalModel.h"
#include "src/physicalModel/FunctionnalModelFactory.h"
#include "src/physicalModel/ShkuratovModel/ShkuratovModel.h"
#include "src/physicalModel/HapkeModel/HapkeVersions/Hapke02Model.h"
#include "src/physicalModel/HapkeModel/HapkeAdapters/FourParamsModel.h"
#include "src/physicalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.h"
#include "src/physicalModel/HapkeModel/HapkeAdapters/SixParamsModel.h"
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
    pt::read_json("../test_shkuratov.json", root);  // Load the json file in this ptree
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("eme"))
    {
        geometries[i*3+1] = stod(v.second.data());
        i++;
    }
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("inc"))
    {
        geometries[i*3+0] = stod(v.second.data());

        i++;
    }
    i = 0;



    for (pt::ptree::value_type& v : root.get_child("phi"))
    {
        geometries[i*3+2] = stod(v.second.data());
        i+=1;
    }

    mat photometries = mat(10000,5);
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("an"))
    {
        photometries(i,0) = stod(v.second.data());
        i++;
    }

    i = 0;
    for (pt::ptree::value_type& v : root.get_child("mu1"))
    {
        photometries(i,1) = stod(v.second.data());
        i++;
    }

    i = 0;
    for (pt::ptree::value_type& v : root.get_child("nu"))
    {
        photometries(i,2) = stod(v.second.data());
        i++;
    }

    i = 0;
    for (pt::ptree::value_type& v : root.get_child("m"))
    {
        photometries(i,3) = stod(v.second.data());
        i++;
    }

    i = 0;
    for (pt::ptree::value_type& v : root.get_child("mu2"))
    {
        photometries(i,4) = stod(v.second.data());
        i++;
    }

    /*
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
        photometries(i,1) = stod(v.second.data()) / 30;
        i++;
    }*/

    double scaling[5] = {1.0,1.5,0.8,1.5,1.5};
    double offset[5] = {0,0,0.2,0,0};

    std::shared_ptr<FunctionalModel> myModel (new ShkuratovModel(geometries, 50, 3, scaling, offset));

    auto *x = new double[5*10000];
    for(unsigned k=0; k<10000; k++){
        for(unsigned j=0; j<5; j++){
            x[k*5+j] = photometries(i,j);
        }
    }


    rowvec y(50);
    auto start = chrono::high_resolution_clock::now();
    for(unsigned k=0; k<1; k++){
        myModel->F(photometries.row(k),y);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << duration.count() << endl;
    y.print();

    delete [] x;
    delete [] geometries;




    return 0;
}