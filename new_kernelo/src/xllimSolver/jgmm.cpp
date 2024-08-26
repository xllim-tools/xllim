#include "jgmm.hpp"

JGMM::JGMM()
{
}

mat JGMM::getPosterior()
{
    return posterior_;
}

void JGMM::train(const mat &x, const mat &y, GLLiMParameters<FullCovariance, FullCovariance> &initial_theta, unsigned kmeans_iteration, unsigned em_iteration, double floor, int verbose)
{
    GLLiMParameterstoJGMM(initial_theta); // transform GLLiM parameters to joint GMM parameters

    mat training_data = join_cols(x, y); // create training data set by concatenating X and Y matrices

    // train the GMM with the training data set

    // gmm_full jgmm;
    // jgmm.set_params(jgmm_means, jgmm_covariances, jgmm_weights);
    unsigned n_gaus = jgmm_.n_gaus();
    bool print_mode = (verbose >= 1);
    posterior_ = mat(training_data.n_cols, n_gaus);

    jgmm_.learn(training_data, n_gaus, maha_dist, keep_existing, kmeans_iteration, em_iteration, floor, print_mode);

    for (unsigned k = 0; k < n_gaus; k++)
    {
        posterior_.col(k) = jgmm_.log_p(training_data, k).t();
    }

    // return the GLLiM from the GMM
    JGMMtoGLLiMParameters(initial_theta);
    // return JGMMtoGLLiMParameters(); // transform joint GMM parameters to GLLiM parameters
}

void JGMM::GLLiMParameterstoJGMM(GLLiMParameters<FullCovariance, FullCovariance> &initial_theta)
{
    L_ = initial_theta.L;
    D_ = initial_theta.D;
    K_ = initial_theta.K;

    // GMM weights
    // Rou = theta.Pi;
    jgmm_.reset(L_ + D_, K_);
    jgmm_.set_hefts(initial_theta.Pi);

    // GMM means
    mat AC(D_, K_);
    for (unsigned k = 0; k < K_; k++)
    {
        AC.col(k) = initial_theta.A.slice(k) * initial_theta.C.col(k);
    }
    mat M = join_cols(initial_theta.C, AC + initial_theta.B);
    jgmm_.set_means(M);

    // GMM Covariances
    cube V(D_ + L_, D_ + L_, K_);
    for (unsigned j = 0; j < K_; j++)
    {
        V.slice(j) = join_cols(
            join_rows(initial_theta.Gamma[j] * mat(L_, L_, fill::eye),
                      initial_theta.Gamma[j] * mat(initial_theta.A.slice(j).t())),
            join_rows(initial_theta.A.slice(j) * initial_theta.Gamma[j],
                      initial_theta.Sigma[j] + initial_theta.A.slice(j) * initial_theta.Gamma[j] * initial_theta.A.slice(j).t()));
    }
    jgmm_.set_fcovs(V);
}

void JGMM::JGMMtoGLLiMParameters(GLLiMParameters<FullCovariance, FullCovariance> &theta)
{
    mat m_x = jgmm_.means.submat(0, 0, L_ - 1, K_ - 1);
    mat m_y = jgmm_.means.submat(L_, 0, L_ + D_ - 1, K_ - 1);

    cube v_xx = jgmm_.fcovs.subcube(0, 0, 0, L_ - 1, L_ - 1, K_ - 1);
    cube v_xx_inv = v_xx;
    v_xx_inv.each_slice([](mat &X)
                        { X = inv(X); });
    cube v_xy = jgmm_.fcovs.subcube(0, L_, 0, L_ - 1, L_ + D_ - 1, K_ - 1);
    cube v_xy_t = jgmm_.fcovs.subcube(L_, 0, 0, L_ + D_ - 1, L_ - 1, K_ - 1);
    cube v_yy = jgmm_.fcovs.subcube(L_, L_, 0, L_ + D_ - 1, L_ + D_ - 1, K_ - 1);

    theta.Pi = jgmm_.hefts;
    theta.C = m_x;

    for (unsigned k = 0; k < K_; k++)
    {
        theta.Gamma[k] = v_xx.slice(k);
        theta.A.slice(k) = v_xy_t.slice(k) * v_xx_inv.slice(k);
        theta.B.col(k) = m_y.col(k) - v_xy_t.slice(k) * v_xx_inv.slice(k) * m_x.col(k);
        theta.Sigma[k] = v_yy.slice(k) - v_xy_t.slice(k) * v_xx_inv.slice(k) * v_xy.slice(k);
    }
}
