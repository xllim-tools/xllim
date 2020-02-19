//
// Created by reverse-proxy on 12‏/2‏/2020.
//


#include <armadillo>
#include <utility>
#include "omp.h"

#include "src/learningModel/GLLiMParameters.h"
#include "src/learningModel/LearningConfig.h"
#include "src/learningModel/Estimators.h"
#include "src/learningModel/Icovariance.h"


#include <iostream>
#include <cstring>
#include <exception>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <fstream>


using namespace std;
namespace pt = boost::property_tree;

using namespace learningModel;
using namespace arma;

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
        photometries(i,1) = stod(v.second.data()) / 30.0;
        i++;
    }


    mat y = mat(10000,50);
    i = 0;
    for (pt::ptree::value_type& v : root.get_child("y"))
    {
        int j = 0;
        for(pt::ptree::value_type& elem : v.second){
            y(i,j) = elem.second.get_value<double>();
            j++;
        }
        i++;
    }
    //for(unsigned t=0; t<10; t++){


        y += (mat(y.n_rows, y.n_cols, fill::randu) * 1/100);
        //y = y.submat(0,0,y.n_rows-1,4);
        //photometries = mat(100, 2, fill::randu);

        mat diag_L = mat(photometries.n_cols, photometries.n_cols, fill::zeros);
        mat T_L = trimatu(mat(photometries.n_cols, photometries.n_cols, fill::randu) + diag_L);
        mat S_L = T_L * T_L.t();
        S_L.diag() += 1;
        arma_rng::set_seed_random();

        mat diag_D = mat(y.n_cols, y.n_cols, fill::zeros);;
        mat T_D = trimatu(mat(y.n_cols, y.n_cols, fill::randu) + diag_D);
        mat S_D = T_D * T_D.t();
        S_D.diag() += 1;
        arma_rng::set_seed_random();

        GLLiMParameters<FullCovariance,FullCovariance> myParams = GLLiMParameters<FullCovariance,FullCovariance>();

        int K = 50;

        myParams.A = cube(y.n_cols, photometries.n_cols, K);
        myParams.B = mat(y.n_cols, K);
        myParams.C = mat(photometries.n_cols, K);
        myParams.Pi = normalise(vec(K, fill::randu), 1);
        myParams.Sigma = std::vector<FullCovariance>(K);
        myParams.Gamma = std::vector<FullCovariance>(K);



        for(unsigned p=0; p < K ; p++ ){
            myParams.Sigma[p] = FullCovariance(S_D);
            myParams.Gamma[p] = FullCovariance(S_L);
            myParams.A.slice(p) = mat(y.n_cols, photometries.n_cols, fill::randu);
            arma_rng::set_seed_random();
            myParams.B.col(p) = vec(y.n_cols, fill::randu);
            arma_rng::set_seed_random();
            myParams.C.col(p) = vec(photometries.n_cols, fill::randu);
            arma_rng::set_seed_random();
        }




        std::shared_ptr<GMMLearningConfig> myLearningconfig (new GMMLearningConfig(0,10));
        GmmEstimator estimator (myLearningconfig);


        auto start = chrono::high_resolution_clock::now();
        estimator.estimate(photometries, y, myParams);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end - start);
        cout << duration.count() << endl;
   // }





}

