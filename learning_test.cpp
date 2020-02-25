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

/*int main(){

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


        y += (mat(y.n_rows, y.n_cols, fill::randn) * 1/100);
        //y = y.submat(0,0,y.n_rows-1,4);
        //photometries = mat(100, 2, fill::randn);

        arma_rng::set_seed_random();
        mat T_L = trimatl(mat(photometries.n_cols, photometries.n_cols, fill::randn));
        mat S_L = T_L * T_L.t();
        S_L.diag() += 1;
        arma_rng::set_seed_random();

        mat T_D = trimatl(mat(y.n_cols, y.n_cols, fill::randn));
        mat S_D = T_D * T_D.t();
        S_D.diag() += 1;
        arma_rng::set_seed_random();

        GLLiMParameters<FullCovariance,FullCovariance> myParams = GLLiMParameters<FullCovariance,FullCovariance>();

        int K = 20;

        myParams.A = cube(y.n_cols, photometries.n_cols, K);
        myParams.B = mat(y.n_cols, K);
        myParams.C = mat(photometries.n_cols, K);
        myParams.Pi = normalise(vec(K, fill::randn), 1);
        myParams.Sigma = std::vector<FullCovariance>(K);
        myParams.Gamma = std::vector<FullCovariance>(K);

        //S_D.print("Sigma");
        //S_L.print("Gamma");


        for(unsigned p=0; p < K ; p++ ){
            myParams.Sigma[p] = FullCovariance(S_D);
            myParams.Gamma[p] = FullCovariance(S_L);

            myParams.A.slice(p) = mat(y.n_cols, photometries.n_cols, fill::randn);
            arma_rng::set_seed_random();
            myParams.B.col(p) = vec(y.n_cols, fill::randn);
            arma_rng::set_seed_random();
            myParams.C.col(p) = vec(photometries.n_cols, fill::randn);
            arma_rng::set_seed_random();
        }




        //std::shared_ptr<GMMLearningConfig> myLearningconfig (new GMMLearningConfig(0,10));
        //GmmEstimator estimator (myLearningconfig);

        std::shared_ptr<EMLearningConfig> myLearningconfig (new EMLearningConfig(10,1));
        EmEstimator<FullCovariance, FullCovariance> estimator(myLearningconfig);

        vec l = ones<vec>(5);
        vec m = ones<vec>(5);
        //l.print("l");
        //m.print("m");
        //std::cout << dot(l, m) << std::endl;



        //auto start = chrono::high_resolution_clock::now();
        estimator.estimate(photometries, y, myParams);
        //auto end = chrono::high_resolution_clock::now();
        //auto duration = chrono::duration_cast<chrono::seconds>(end - start);
        //cout << duration.count() << endl;
   // }





}*/

int main(){

    int L = 6;
    int D = 50;
    int N = 10000;
    int K = 3;

    // Fixer A et B
    arma_rng::set_seed(10000);
    mat A(D,L, fill::randu);
    arma_rng::set_seed(11000);
    vec B(D, fill::randu);

    // Fixer sigma
    mat sigma(N,D, fill::randn);
    sigma *= sqrt(0.01);

    // Fixer C, Gamma et Pi
    arma_rng::set_seed(11100);
    vec Pi = normalise(vec(K, fill::randu), 1);

    arma_rng::set_seed(11110);
    mat C(L,K, fill::randu);

    cube Gamma(L,L,K);
    arma_rng::set_seed(11111);
    mat T_L = trimatl(mat(L, L, fill::randu));
    Gamma.slice(0) = T_L * T_L.t();
    Gamma.slice(0).diag() += 1;

    arma_rng::set_seed(22222);
    T_L = trimatl(mat(L, L, fill::randu));
    Gamma.slice(1) = T_L * T_L.t();
    Gamma.slice(1).diag() += 1;

    arma_rng::set_seed(33333);
    T_L = trimatl(mat(L, L, fill::randu));
    Gamma.slice(2) = T_L * T_L.t();
    Gamma.slice(2).diag() += 1;

    // init gmm
    gmm_full model;
    model.set_params(C, Gamma, Pi.t());

    // sample X
    mat X = model.generate(N);

    // Calculer Y
    mat Y = mat(A * X);
    for(unsigned n=0; n<N; n++){
        for(unsigned d=0; d<D; d++){
            Y(d,n) += B(d) + sigma(n,d);
        }

    }



    /*mat u(D,D,fill::zeros);
    for(unsigned n=0; n<N; n++){
        mat temp = mat(Y.col(n) - A * X.col(n) - B);

        u = u + temp * temp.t();
    }

    u /= N;
    //std::cout << sum(u) << std::endl;
    u.print();*/

    arma_rng::set_seed_random();
    T_L = trimatl(mat(L, L, fill::randu));
    mat S_L = T_L * T_L.t();
    S_L.diag() += 1;
    arma_rng::set_seed_random();

    mat T_D = trimatl(mat(D, D, fill::randu));
    mat S_D = T_D * T_D.t();
    S_D.diag() += 1;
    arma_rng::set_seed_random();


    GLLiMParameters<FullCovariance,FullCovariance> myParams = GLLiMParameters<FullCovariance,FullCovariance>();
    myParams.A = cube(D, L, K);
    myParams.B = mat(D, K);
    myParams.C = mat(L, K);
    myParams.Pi = normalise(vec(K, fill::randu), 1);
    myParams.Sigma = std::vector<FullCovariance>(K);
    myParams.Gamma = std::vector<FullCovariance>(K);

    mat sig(D,D, fill::zeros);
    sig.diag() += 0.01;
    //sig.print();

    for(unsigned p=0; p < K ; p++ ){

       /* myParams.Sigma[p] = FullCovariance(sig);
        myParams.Gamma[p] = FullCovariance(Gamma.slice(p));
        myParams.A.slice(p) = A;
        myParams.B.col(p) = B;
        myParams.C.col(p) = C.col(p);*/

        myParams.Sigma[p] = FullCovariance(S_D);
        myParams.Gamma[p] = FullCovariance(S_L);

        myParams.A.slice(p) = mat(D,L, fill::randu);
        arma_rng::set_seed_random();
        myParams.B.col(p) = vec(D, fill::randu);
        arma_rng::set_seed_random();
        myParams.C.col(p) = vec(L, fill::randu);
        arma_rng::set_seed_random();
    }

    myParams.A.print("Init A");
    myParams.B.t().print("Init B");
    myParams.C.print("Init C");
    S_D.print("init Sigma");
    S_L.print("init Gamma");
    myParams.Pi.print("Init Pi");


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

    /*std::shared_ptr<GMMLearningConfig> myLearningconfig (new GMMLearningConfig(0,10));
    GmmEstimator estimator (myLearningconfig);*/

    y += (mat(y.n_rows, y.n_cols, fill::randn) * 1/100);

    std::shared_ptr<EMLearningConfig> myLearningconfig (new EMLearningConfig(10,1));
    EmEstimator<FullCovariance, FullCovariance> estimator(myLearningconfig);


    estimator.estimate(photometries, y, make_shared<GLLiMParameters<FullCovariance,FullCovariance>>(myParams));







}

