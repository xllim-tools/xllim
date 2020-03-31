//
// Created by reverse-proxy on 13‏/2‏/2020.
//


#include "GLLiMLearning.h"

using namespace learningModel;

template<typename T, typename U>
GLLiMLearning<T, U>::GLLiMLearning(std::shared_ptr<Iinitilizer<T, U>> initializer,
                                   std::shared_ptr<Iestimator<T, U>> estimator, unsigned K) {
    this->initializer = initializer;
    this->estimator = estimator;
    this->K = K;
}

template<typename T, typename U>
void GLLiMLearning<T, U>::initialize(const mat &x, const mat &y) {
    this->gllim_parameters = this->initializer->execute(x, y, this->K);
}

template<typename T, typename U>
void GLLiMLearning<T, U>::train(const mat &x, const mat &y) {
    this->estimator->execute(x,y,this->gllim_parameters);
}

template<typename T, typename U>
void GLLiMLearning<T, U>::exportModel(GLLiM &gllim) {
    if(this->gllim_parameters->K == gllim.K &&
            this->gllim_parameters->D == gllim.D &&
            this->gllim_parameters->L == gllim.L){

        for(unsigned k=0; k<gllim.K; k++){
            gllim.Pi[k] = this->gllim_parameters->Pi(k);

            mat arma_gamma = this->gllim_parameters->Gamma[k].getFull();
            for(unsigned l=0; l<gllim.L; l++){
                gllim.C[l + k*gllim.L] = this->gllim_parameters->C(l,k);
                for(unsigned l2=0; l2<gllim.L; l2++){

                    gllim.Gamma[l2 + l*gllim.L + k*gllim.L*gllim.L] = arma_gamma(l2,l);
                }
            }
            mat arma_sigma = this->gllim_parameters->Sigma[k].getFull();
            for(unsigned d=0; d<gllim.D; d++){
                gllim.B[d + k*gllim.D] = this->gllim_parameters->B(d,k);
                for(unsigned d2=0; d2<gllim.D; d2++){
                    gllim.Sigma[d2 + d*gllim.D + k*gllim.D*gllim.D] = arma_sigma(d2,d);
                }

                for(unsigned l=0; l<gllim.L; l++){
                    gllim.A[d + l*gllim.L + k*gllim.L*gllim.D] = gllim_parameters->A(d,l,k);
                }
            }
        }
    }
}

template<typename T, typename U>
void GLLiMLearning<T, U>::importModel(GLLiM &gllim) {

    gllim_parameters = std::make_shared<GLLiMParameters<T,U>>(gllim.D, gllim.L, gllim.K);
    mat arma_gamma(gllim.L,gllim.L);
    mat arma_sigma(gllim.D,gllim.D);

    for(unsigned k=0; k<gllim.K; k++){
        gllim_parameters->Pi(k) = gllim.Pi[k];

        for(unsigned l=0; l<gllim.L; l++){
            gllim_parameters->C(l,k) = gllim.C[l + k*gllim.L];
            for(unsigned l2=0; l2<gllim.L; l2++){
                arma_gamma(l2,l) = gllim.Gamma[l2 + l*gllim.L + k*gllim.L*gllim.L];
            }
        }
        gllim_parameters->Gamma[k] = arma_gamma;

        for(unsigned d=0; d<gllim.D; d++){
            gllim_parameters->B(d,k) = gllim.B[d + k*gllim.D];
            for(unsigned d2=0; d2<gllim.D; d2++){
                arma_sigma(d2,d) = gllim.Sigma[d2 + d*gllim.D + k*gllim.D*gllim.D];
            }
            for(unsigned l=0; l<gllim.L; l++){
                gllim_parameters->A(d,l,k) = gllim.A[d + l*gllim.L + k*gllim.L*gllim.D];
            }
        }
        gllim_parameters->Sigma[k] = arma_sigma;
    }
}


template<typename T, typename U>
GLLiMParameters<FullCovariance,FullCovariance> GLLiMLearning<T, U>::inverse(GLLiMParameters<T,U> &gllim_direct) {

    GLLiMParameters<FullCovariance, FullCovariance> gllim_inv(gllim_direct.L, gllim_direct.D, gllim_direct.K);

    for(unsigned k=0; k<gllim_direct.K; k++){
        if(gllim_direct.Pi(k) != 0){
            gllim_inv.Pi(k) = gllim_direct.Pi(k);
            FullCovariance sigma_inv = FullCovariance(gllim_direct.Sigma[k].inv().getFull());
            FullCovariance gamma_inv = FullCovariance(gllim_direct.Gamma[k].inv().getFull());
            gllim_inv.C.col(k) = gllim_direct.A.slice(k) * gllim_direct.C.col(k) + gllim_direct.B.col(k);
            gllim_inv.Gamma[k] = FullCovariance(gllim_direct.Sigma[k] + gllim_direct.A.slice(k) * gllim_direct.Gamma[k] * gllim_direct.A.slice(k).t());
            gllim_inv.Sigma[k] = FullCovariance(gamma_inv + mat(gllim_direct.A.slice(k).t()) * sigma_inv * mat(gllim_direct.A.slice(k))).inv();
            gllim_inv.A.slice(k) = gllim_inv.Sigma[k] * mat(gllim_direct.A.slice(k).t()) * sigma_inv;
            gllim_inv.B.col(k) = gllim_inv.Sigma[k] * vec(gamma_inv * vec(gllim_direct.C.col(k)) - mat(gllim_direct.A.slice(k).t()) * sigma_inv * gllim_direct.B.col(k));
        }
    }
    return gllim_inv;
}

template<typename T, typename U>
arma::gmm_full GLLiMLearning<T, U>::computeGMM(const vec &y_obs, const vec &cov_obs) {

    // compute P_X|Y=y which is a GMM with weights , means and covariances deduced from the inversed GLLiM

    // 1 - alter sigma covariance
    GLLiMParameters<T, U> temp_gllim = *gllim_parameters;
    this->alterCovariance(temp_gllim, cov_obs);

    // 2 - inverse theta_obs
    GLLiMParameters<FullCovariance, FullCovariance> gllim_inv = inverse(temp_gllim);

    // 3 - construct the GMM
    return this->logDensity(gllim_inv, y_obs);
}

template<typename T, typename U>
template <typename V, typename W>
arma::gmm_full GLLiMLearning<T, U>::logDensity(GLLiMParameters<V,W> &gllim, const vec &x) {
    static_assert(std::is_base_of<Icovariance, V>(), "Type V must be Icovariance specialization");
    static_assert(std::is_base_of<Icovariance, W>(), "Type W must be Icovariance specialization");

    gmm_full model;

    vec weights(gllim.K,fill::zeros);
    for(unsigned k=0; k<gllim.K; k++){
        double det_gamma = gllim.Gamma[k].det();
        vec x_u = x - gllim.C.col(k);
        if(det_gamma != 0){
            weights(k) = log(gllim.Pi(k)) - 0.5 * (gllim.L * log(2* datum::pi) + log(det_gamma) + dot((rowvec(x_u.t()) * gllim.Gamma[k].inv()).t(), x_u));
        }
    }

    double result = 0;
    double max = weights.max();
    if(max != -datum::inf){
        for(unsigned k=0; k<gllim.K; k++){
            result += exp(weights(k) - max);
        }
        result = log(result) + max;
    }
    if(result != -datum::inf){
        weights = exp(weights - result);
    }

    // means
    mat means(gllim.D,gllim.K);
    for(unsigned k=0; k<gllim.K; k++){
        means.col(k) = gllim.A.slice(k) * x + gllim.B.col(k);
    }

    // covariances
    cube covariances(gllim.D,gllim.D,gllim.K);
    for(unsigned k=0; k<gllim.K; k++){
        covariances.slice(k) = gllim.Sigma[k].getFull();
    }


    model.set_params(means,covariances,weights.t());
    return model;
}

template<typename T, typename U>
void GLLiMLearning<T,U>::alterCovariance(GLLiMParameters<T, U> &gllim, const vec &cov_obs){
    mat cov(gllim.D, gllim.K , fill::zeros);
    cov.diag() += cov_obs;
    for(unsigned k=0; k<gllim.K; k++){
        gllim.Sigma[k] += cov;
    }
}




