/**
 * @file GmmEstimator.cpp
 * @brief GmmEstimator class implementation
 * @author Sami DJOUADI
 * @version 1.1
 * @date 13/02/2020
 */

using namespace learningModel;

GmmEstimator::GmmEstimator(const std::shared_ptr<GMMLearningConfig> &config) {
    this->config = config;
}

GmmEstimator::GmmEstimator() {
    this->config = std::make_shared<GMMLearningConfig>(GMMLearningConfig());
}


mat GmmEstimator::getPosterior() {
    return posterior;
}

void GmmEstimator::train(const mat &data, const vec& weights, const mat &means, const cube &covariances){
    gmm_full model;
    unsigned int n_gaus = weights.n_rows;
    posterior = mat(data.n_cols, n_gaus);
    model.set_params(means, covariances, weights.t());
    if(config->em_iteration == 0){
        for(unsigned k=0; k<n_gaus; k++){
            posterior.col(k) = model.log_p(data,k).t();
        }
    }else{
        if(model.learn(data, n_gaus, maha_dist, keep_existing, config->kmeans_iteration, config->em_iteration,config->floor ,false)){
            for(unsigned k=0; k<n_gaus; k++){
                posterior.col(k) = model.log_p(data,k).t();
            }
        }
        //else
        //    throw std::string("GMM learning failed");
    }


}

void GmmEstimator::execute(const arma::mat & x, const arma::mat & y,
        std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance> > initial_theta) {
    // transform GLLiM parameters to GMM parameters
    this->toGMM(initial_theta);

    // create training data set by concatenating X and Y matrices
    mat training_data = join_cols(x.t(),y.t());

    // train the GMM with the training data set
    train(x,Rou,M,V);

    // return the GLLiM from the GMM
    initial_theta->operator=(fromGMM(Rou.n_rows, y.n_cols, x.n_cols));
}

void GmmEstimator::toGMM(const std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>>& theta) {
    unsigned int K = theta->Pi.n_rows;
    unsigned int L = theta->C.n_rows;
    unsigned int D = theta->B.n_rows;

    // GMM weights
    this->Rou = theta->Pi;

    // GMM means
    mat AC = mat(D,K);
    for(unsigned i = 0; i < K; i++){
        AC.col(i) = theta->A.slice(i) * theta->C.col(i);
    }
    this->M = join_cols(theta->C, AC + theta->B);

    // GMM Covariances
    V = cube(D+L, D+L, K);
    for(unsigned j = 0; j < K; j++){
        V.slice(j) =join_cols(
                join_rows(theta->Gamma[j] * mat(L, L, fill::eye),
                          theta->Gamma[j] * mat(theta->A.slice(j).t())),
                join_rows(theta->A.slice(j) * theta->Gamma[j],
                          theta->Sigma[j] + theta->A.slice(j) * theta->Gamma[j] * theta->A.slice(j).t()));
    }
}

GLLiMParameters<FullCovariance, FullCovariance> GmmEstimator::fromGMM(unsigned int K, unsigned int D, unsigned int L) {
    GLLiMParameters<FullCovariance, FullCovariance> gLLiMParameters(D,L,K);

    mat m_x = M.submat(0, 0, L-1, K-1);
    mat m_y = M.submat(L, 0, L+D-1, K-1);

    cube v_xx = V.subcube(0, 0, 0, L-1, L-1, K-1);
    cube v_xx_inv = v_xx;
    v_xx_inv.each_slice( [](mat& X){X = inv(X); } );
    cube v_xy = V.subcube(0, L, 0, L-1, L+D-1, K-1);
    cube v_xy_t = V.subcube(L, 0, 0, L+D-1, L-1, K-1);
    cube v_yy = V.subcube(L, L, 0, L+D-1, L+D-1, K-1);

    gLLiMParameters.Pi = this->Rou;
    gLLiMParameters.C = m_x;

    for(unsigned i = 0; i < K; i++){
        gLLiMParameters.Gamma[i] = v_xx.slice(i);
        gLLiMParameters.A.slice(i) = v_xy_t.slice(i) * v_xx_inv.slice(i);
        gLLiMParameters.B.col(i) = m_y.col(i) - v_xy_t.slice(i) * v_xx_inv.slice(i) * m_x.col(i);
        gLLiMParameters.Sigma[i] = v_yy.slice(i) - v_xy_t.slice(i) * v_xx_inv.slice(i) * v_xy.slice(i);
    }
    return gLLiMParameters;
}


