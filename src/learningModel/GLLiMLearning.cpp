//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#include "GLLiMLearning.h"

using namespace learningModel;

template<typename T, typename U>
GLLiMLearning<T, U>::GLLiMLearning(std::shared_ptr<Iinitilizer<T, U>> initializer,
                                   std::shared_ptr<Iestimator<T, U>> estimator,
                                   unsigned gaussians) {
    this->initializer = initializer;
    this->estimator = estimator;
    this->K = gaussians;
}

template<typename T, typename U>
void GLLiMLearning<T, U>::initialize(const mat &x, const mat &y) {
    this->gllim_parameters = this->initializer->execute(x, y, this->K);
}

template<typename T, typename U>
void GLLiMLearning<T, U>::train(const mat &x, const mat &y) {
    this->estimator->execute(x,y,this->gllim_parameters);
}
