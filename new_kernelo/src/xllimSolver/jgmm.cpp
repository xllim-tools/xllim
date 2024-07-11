#include "jgmm.hpp"

JGMM::JGMM()
{
}

mat JGMM::getPosterior()
{
    return this->posterior;
}

GLLiMParameters<FullCovariance,FullCovariance> JGMM::train(const mat &x, const mat &y, GLLiMParameters<FullCovariance,FullCovariance> &initial_theta, unsigned kmeans_iteration, unsigned em_iteration, double floor)
{
    this->GLLiMParameterstoJGMM(initial_theta); // transform GLLiM parameters to joint GMM parameters

    mat training_data = join_cols(x.t(), y.t()); // create training data set by concatenating X and Y matrices

    // train the GMM with the training data set

    // gmm_full jgmm;
    // jgmm.set_params(this->jgmm_means, this->jgmm_covariances, this->jgmm_weights);
    unsigned n_gaus = this->jgmm.n_gaus();
    posterior = mat(training_data.n_cols, n_gaus);

    this->jgmm.learn(training_data, n_gaus, maha_dist, keep_existing, kmeans_iteration, em_iteration, floor, true);

    for (unsigned k = 0; k < n_gaus; k++)
    {
        posterior.col(k) = this->jgmm.log_p(training_data, k).t();
    }

    // return the GLLiM from the GMM
    return this->JGMMtoGLLiMParameters(); // transform joint GMM parameters to GLLiM parameters
}

void JGMM::GLLiMParameterstoJGMM(GLLiMParameters<FullCovariance,FullCovariance> &initial_theta)
{
    this->L = initial_theta.L;
    this->D = initial_theta.D;
    this->K = initial_theta.K;

    // GMM weights
    // this->Rou = theta.Pi;
    this->jgmm.reset(L + D, K);
    std::cout << "K=" << std::to_string(K) << std::endl;
    std::cout << "n_dims=" << std::to_string(this->jgmm.n_dims()) << std::endl;
    std::cout << "n_gaus=" << std::to_string(this->jgmm.n_gaus()) << std::endl;
    this->jgmm.set_hefts(initial_theta.Pi);

    // GMM means
    mat AC(D, K);
    for (unsigned k = 0; k < K; k++)
    {
        AC.col(k) = initial_theta.A.slice(k) * initial_theta.C.col(k);
    }
    mat M = join_cols(initial_theta.C, AC + initial_theta.B);
    std::cout<< std::to_string(M.n_rows) << std::to_string(M.n_cols) << std::endl;
    this->jgmm.set_means(M);

    // GMM Covariances
    cube V(D + L, D + L, K);
    for (unsigned j = 0; j < K; j++)
    {
        V.slice(j) = join_cols(
            join_rows(initial_theta.Gamma[j] * mat(L, L, fill::eye),
                      initial_theta.Gamma[j] * mat(initial_theta.A.slice(j).t())),
            join_rows(initial_theta.A.slice(j) * initial_theta.Gamma[j],
                      initial_theta.Sigma[j] + initial_theta.A.slice(j) * initial_theta.Gamma[j] * initial_theta.A.slice(j).t()));
    }
    std::cout<< std::to_string(V.n_rows) << std::to_string(V.n_cols) << std::to_string(V.n_slices) << std::endl;
    this->jgmm.set_fcovs(V);
}

GLLiMParameters<FullCovariance,FullCovariance> JGMM::JGMMtoGLLiMParameters()
{
    GLLiMParameters<FullCovariance,FullCovariance> theta(L, D, K);

    mat m_x = this->jgmm.means.submat(0, 0, L - 1, K - 1);
    mat m_y = this->jgmm.means.submat(L, 0, L + D - 1, K - 1);

    cube v_xx = this->jgmm.fcovs.subcube(0, 0, 0, L - 1, L - 1, K - 1);
    cube v_xx_inv = v_xx;
    v_xx_inv.each_slice([](mat &X)
                        { X = inv(X); });
    cube v_xy = this->jgmm.fcovs.subcube(0, L, 0, L - 1, L + D - 1, K - 1);
    cube v_xy_t = this->jgmm.fcovs.subcube(L, 0, 0, L + D - 1, L - 1, K - 1);
    cube v_yy = this->jgmm.fcovs.subcube(L, L, 0, L + D - 1, L + D - 1, K - 1);

    theta.Pi = this->jgmm.hefts;
    theta.C = m_x;

    for (unsigned k = 0; k < K; k++)
    {
        theta.Gamma[k] = v_xx.slice(k);
        theta.A.slice(k) = v_xy_t.slice(k) * v_xx_inv.slice(k);
        theta.B.col(k) = m_y.col(k) - v_xy_t.slice(k) * v_xx_inv.slice(k) * m_x.col(k);
        theta.Sigma[k] = v_yy.slice(k) - v_xy_t.slice(k) * v_xx_inv.slice(k) * v_xy.slice(k);
    }
    return theta;
}
