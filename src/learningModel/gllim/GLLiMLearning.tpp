//
// Created by reverse-proxy on 13‏/2‏/2020.
//


using namespace learningModel;

template<typename T, typename U>
GLLiMLearning<T, U>::GLLiMLearning(std::shared_ptr<Iinitilizer<T, U>> initializer,
                                   std::shared_ptr<Iestimator<T, U>> estimator, unsigned gaussians) {
    this->initializer = initializer;
    this->estimator = estimator;
    nb_gaussians = gaussians;
}

template<typename T, typename U>
void GLLiMLearning<T, U>::initialize(const mat &x, const mat &y) {
    this->gllim_parameters = this->initializer->execute(x, y, this->nb_gaussians);
}

template<typename T, typename U>
void GLLiMLearning<T, U>::train(const mat &x, const mat &y) {
    this->estimator->execute(x,y,this->gllim_parameters);

    this->gllim_parameters->Pi.print("Pi");
    this->gllim_parameters->B.print("B");
    this->gllim_parameters->C.print("C");
    this->gllim_parameters->A.print("A");
    this->gllim_parameters->Gamma[0].print();
    this->gllim_parameters->Sigma[0].print();

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

        ;
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
    unsigned non_null_weights = 0;
    for(unsigned k=0; k<gllim_direct.K; k++)
        if(gllim_direct.Pi(k) != 0)
            non_null_weights++;

    GLLiMParameters<FullCovariance, FullCovariance> gllim_inv(gllim_direct.L, gllim_direct.D, non_null_weights);

    unsigned i =0;
    for(unsigned k=0; k<gllim_direct.K; k++){
        if(gllim_direct.Pi(k) != 0){
            gllim_inv.Pi(i) = gllim_direct.Pi(i);
            FullCovariance sigma_inv = FullCovariance(gllim_direct.Sigma[i].inv().getFull());
            FullCovariance gamma_inv = FullCovariance(gllim_direct.Gamma[i].inv().getFull());
            gllim_inv.C.col(i) = gllim_direct.A.slice(i) * gllim_direct.C.col(i) + gllim_direct.B.col(i);
            gllim_inv.Gamma[i] = FullCovariance(gllim_direct.Sigma[i] + gllim_direct.A.slice(i) * gllim_direct.Gamma[i] * gllim_direct.A.slice(i).t());
            gllim_inv.Sigma[i] = FullCovariance(gamma_inv + mat(gllim_direct.A.slice(i).t()) * sigma_inv * mat(gllim_direct.A.slice(i))).inv();
            gllim_inv.A.slice(i) = gllim_inv.Sigma[i] * mat(gllim_direct.A.slice(i).t()) * sigma_inv;
            gllim_inv.B.col(i) = gllim_inv.Sigma[i] * vec(gamma_inv * vec(gllim_direct.C.col(i)) - mat(gllim_direct.A.slice(i).t()) * sigma_inv * gllim_direct.B.col(i));
            i++;
        }
    }
    return gllim_inv;
}

template<typename T, typename U>
arma::gmm_full GLLiMLearning<T, U>::computeGMM(const vec &y_obs, const vec &cov_obs) {

    // compute P_X|Y=y which is a GMM with weights , means and covariances deduced from the inversed GLLiM

    // 1 - alter sigma covariance
    GLLiMParameters<T, U> temp_gllim = *gllim_parameters;
    mat cov(temp_gllim.D, temp_gllim.K , fill::zeros);
    cov.diag() += cov_obs;
    for(unsigned k=0; k<temp_gllim.K; k++){
        temp_gllim.Sigma[k] += cov;
    }

    // 2 - inverse theta_obs
    GLLiMParameters<FullCovariance, FullCovariance> gllim_inv = inverse(temp_gllim);

    // 3 - construct the GMM

    // weights
    vec weights(gllim_inv.K,fill::zeros);
    for(unsigned k=0; k<gllim_inv.K; k++){
        double det_gamma = gllim_inv.Gamma[k].det();
        vec y_u = y_obs - gllim_inv.C.col(k);
        if(det_gamma != 0){
            weights(k) = log(gllim_inv.Pi(k)) - 0.5 * (gllim_inv.L * log(2* datum::pi) + log(det_gamma) + dot((rowvec(y_u.t()) * gllim_inv.Gamma[k].inv()).t(), y_u));
        }
    }

    double result = 0;
    double max = weights.max();
    if(max != -datum::inf){
        for(unsigned k=0; k<gllim_inv.K; k++){
            result += exp(weights(k) - max);
        }
        result = log(result) + max;
    }
    if(result != -datum::inf){
        weights = exp(weights - result);
    }

    // means
    mat means(gllim_inv.D,gllim_inv.K);
    for(unsigned k=0; k<gllim_inv.K; k++){
        means.col(k) = gllim_inv.A.slice(k) * y_obs + gllim_inv.B.col(k);
    }

    // covariances
    cube covariances(gllim_inv.D,gllim_inv.D,gllim_inv.K);
    for(unsigned k=0; k<gllim_inv.K; k++){
        covariances.slice(k) = gllim_inv.Sigma[k].getFull();
    }

    gmm_full model;
    model.set_params(means,covariances,weights.t());

    return model;
}




