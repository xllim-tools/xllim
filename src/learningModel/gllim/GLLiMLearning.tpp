/**
 * @file GLLiMLearning.tpp
 * @brief GLLiMLearning class implementation
 * @author Sami DJOUADI
 * @version 1.1
 * @date 13/02/2020
 */


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
    this->inverse_gllim_parameters = std::make_shared<GLLiMParameters<FullCovariance, FullCovariance>>(
            this->inverse(*gllim_parameters)
            );
}

template<typename T, typename U>
void GLLiMLearning<T, U>::getModel(GLLiM &gllim) {
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
            }
        }
        for(unsigned i=0; i<gllim.L * gllim.D * gllim.K; i++){
            gllim.A[i] = gllim_parameters->A((i % (gllim.L * gllim.D))% (gllim.L),
                           (i % (gllim.L * gllim.D))/ (gllim.L),
                           i / (gllim.L * gllim.D));
        }
    }
}

template<typename T, typename U>
void GLLiMLearning<T, U>::setModel(GLLiM &gllim) {

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
        }
        gllim_parameters->Sigma[k] = arma_sigma;
    }

    for(unsigned i=0; i<gllim.L * gllim.D * gllim.K; i++){
        gllim_parameters->A((i % (gllim.L * gllim.D))% (gllim.L),
                                         (i % (gllim.L * gllim.D))/ (gllim.L),
                                         i / (gllim.L * gllim.D)) = gllim.A[i];
    }

    this->inverse_gllim_parameters = std::make_shared<GLLiMParameters<FullCovariance, FullCovariance>>(
            this->inverse(*gllim_parameters)
    );
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
            gllim_inv.B.col(k) = gllim_inv.Sigma[k] * vec(gamma_inv * vec(gllim_direct.C.col(k)) - mat(gllim_direct.A.slice(k).t()) * sigma_inv * vec(gllim_direct.B.col(k)));
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
    return this->logDensity(std::make_shared<GLLiMParameters<FullCovariance, FullCovariance>>(gllim_inv), y_obs);
}

template<typename T, typename U>
template <typename V, typename W>
arma::gmm_full GLLiMLearning<T, U>::logDensity(std::shared_ptr<GLLiMParameters<V,W>> gllim, const vec &x) {
    static_assert(std::is_base_of<Icovariance, V>(), "Type V must be Icovariance specialization");
    static_assert(std::is_base_of<Icovariance, W>(), "Type W must be Icovariance specialization");

    gmm_full model;
    double log_det_gamma;
    vec x_u;

    vec weights(gllim->K,fill::zeros);
    for(unsigned k=0; k<gllim->K; k++){
        if(gllim->Pi(k) == 0){
            weights(k) = -datum::inf;
        }else{
            log_det_gamma = gllim->Gamma[k].log_det();
            x_u = x - gllim->C.col(k);
            if(log_det_gamma != -datum::inf){
                weights(k) = log(gllim->Pi(k)) - 0.5 * (gllim->L * log(2* datum::pi) + log_det_gamma + dot((rowvec(x_u.t()) * gllim->Gamma[k].inv()).t(), x_u));
            }
        }
    }

    double result = 0;
    double max = weights.max();
    if(max != -datum::inf){
        for(unsigned k=0; k<gllim->K; k++){
            result += exp(weights(k) - max);
        }
        result = log(result) + max;
    }
    if(result != -datum::inf){
        weights = exp(weights - result);
    }


    // means
    mat means(gllim->D,gllim->K);
    for(unsigned k=0; k<gllim->K; k++){
        means.col(k) = gllim->A.slice(k) * x + gllim->B.col(k);
    }

    // covariances
    cube covariances(gllim->D,gllim->D,gllim->K);
    for(unsigned k=0; k<gllim->K; k++){
        covariances.slice(k) = gllim->Sigma[k].getFull();
    }


    model.set_params(means,covariances,weights.t());
    return model;
}

template<typename T, typename U>
void GLLiMLearning<T,U>::alterCovariance(GLLiMParameters<T, U> &gllim, const vec &cov_obs){
    mat cov(gllim.D, gllim.K , fill::zeros);
    cov.diag() += cov_obs;
    for(unsigned k=0; k<gllim.K; k++){
        gllim.Sigma[k] += pow(cov, 2);
    }
}

template<typename T, typename U>
void GLLiMLearning<T, U>::getInverse(GLLiM &gllim) {
    if(inverse_gllim_parameters->K == gllim.K &&
       inverse_gllim_parameters->D == gllim.D &&
       inverse_gllim_parameters->L == gllim.L){

        for(unsigned k=0; k<gllim.K; k++){
            gllim.Pi[k] = inverse_gllim_parameters->Pi(k);

            mat arma_gamma = inverse_gllim_parameters->Gamma[k].getFull();
            for(unsigned l=0; l<gllim.L; l++){
                gllim.C[l + k*gllim.L] = inverse_gllim_parameters->C(l,k);
                for(unsigned l2=0; l2<gllim.L; l2++){

                    gllim.Gamma[l2 + l*gllim.L + k*gllim.L*gllim.L] = arma_gamma(l2,l);
                }
            }
            mat arma_sigma = inverse_gllim_parameters->Sigma[k].getFull();
            for(unsigned d=0; d<gllim.D; d++){
                gllim.B[d + k*gllim.D] = inverse_gllim_parameters->B(d,k);
                for(unsigned d2=0; d2<gllim.D; d2++){
                    gllim.Sigma[d2 + d*gllim.D + k*gllim.D*gllim.D] = arma_sigma(d2,d);
                }
            }
        }

        for(unsigned i=0; i<gllim.L * gllim.D * gllim.K; i++){
            gllim.A[i] = inverse_gllim_parameters->A((i % (gllim.L * gllim.D))% (gllim.L),
                                             (i % (gllim.L * gllim.D))/ (gllim.L),
                                             i / (gllim.L * gllim.D));
        }
    }
}

template<typename T, typename U>
void GLLiMLearning<T, U>::directLogDensity(double *x, double *weights, double *means, double *covs) {
    vec x_obs(&x[0], gllim_parameters->L);

    gmm_full gmm = logDensity(gllim_parameters, x_obs);

    for(unsigned k=0; k<gllim_parameters->K; k++){
        weights[k] = gmm.hefts(k);
        for(unsigned d=0; d<gllim_parameters->D; d++){
            means[d + k * gllim_parameters->D] = gmm.means(d,k);
        }
    }

    for(unsigned i=0; i<gllim_parameters->D * gllim_parameters->D * gllim_parameters->K; i++){
        covs[i] = gmm.fcovs(
                (i % (gllim_parameters->D * gllim_parameters->D))% (gllim_parameters->D),
                (i % (gllim_parameters->D * gllim_parameters->D))/ (gllim_parameters->D),
                i / (gllim_parameters->D * gllim_parameters->D)
                );
    }
}

template<typename T, typename U>
void GLLiMLearning<T, U>::inverseLogDensity(double *y, double *weights, double *means, double *covs) {
    vec y_obs(&y[0], inverse_gllim_parameters->L);

    gmm_full gmm = logDensity(inverse_gllim_parameters, y_obs);

    for(unsigned k=0; k<inverse_gllim_parameters->K; k++){
        weights[k] = gmm.hefts(k);
        for(unsigned d=0; d<inverse_gllim_parameters->D; d++){
            means[d + k * inverse_gllim_parameters->D] = gmm.means(d,k);
        }
    }

    for(unsigned i=0; i<inverse_gllim_parameters->D * inverse_gllim_parameters->D * inverse_gllim_parameters->K; i++){
        covs[i] = gmm.fcovs(
                (i % (inverse_gllim_parameters->D * inverse_gllim_parameters->D))% (inverse_gllim_parameters->D),
                (i % (inverse_gllim_parameters->D * inverse_gllim_parameters->D))/ (inverse_gllim_parameters->D),
                i / (inverse_gllim_parameters->D * inverse_gllim_parameters->D)
        );
    }

}






