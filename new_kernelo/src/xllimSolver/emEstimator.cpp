#include "omp.h"
#include "emEstimator.hpp"
#include "../utils/utils.hpp"
#include "../logging/logger.hpp"

#define LOG_2_PI log(2 * datum::pi)

template <typename TGamma, typename TSigma>
EmEstimator<TGamma, TSigma>::EmEstimator() {}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::train(const mat &t, const mat &y, GLLiMParameters<TGamma, TSigma> &theta, unsigned max_iteration, double ratio_ll, double floor, int verbose)
{
    mat log_r(t.n_cols, theta.Pi.n_cols, fill::value(-datum::inf)); // Posterior log probability (N, K)
    // μw = np.zeros((self.Lw, self.N, self.K))
    // Sw = np.zeros((self.Lw, self.Lw, self.K))
    // cube mu_w(this->L_w, t.n_cols, theta.Pi.n_cols); // Gaussian mean of posterior probability r_W|Z (L_w, N, K)
    // cube S_w(this->L_w, this->L_w, theta.Pi.n_cols); // Gaussian covariance matrix of posterior probability r_W|Z (L_w, L_w, K)

    // mat x_t = x.t();
    // mat y_t = y.t();

    // double old_log_likelihood;
    // double new_log_likelihood = -datum::inf;
    unsigned iteration = 0;
    this->log_likelihood = vec(max_iteration + 1, fill::value(-datum::inf)); // on fait une liste des log_ll pour garder ces données en mémoire et pouvoir les retrouver avec insights()
    // double old_log_likelihood;
    // this->log_likelihood(iteration) = -datum::inf; // new_log_likelihood

    // set a logger with a progress bar for GLLiM-EM training
    Logger &logger = Logger::getInstance();
    if (verbose >= 2)
    {
        logger.startProgressBar(max_iteration);
    }
    if (verbose >= 1)
    {
        logger.log(INFO, "Start GLLiM-EM Training");
    }
    do
    {
        iteration++;
        // pyGLLiM
        // # MAXIMIZATION STEP
        // θ = self._maximization(t, y, r, cstr, μw, Sw)

        // # EXPECTATION STEP
        // r, log_like[it], ec = self._expectation_z(t,y,θ)
        // θ, cstr = self._remove_empty_clusters(θ,cstr,ec)
        // μw, Sw = self._expectation_w(t, y, θ)

        // old_log_likelihood = new_log_likelihood;

        // expectation_W_step(t, y, theta, r);

        this->expectation_Z_step(t, y, theta, log_r); // on veut un void ave shared ptr sur theta ou bien un truc qui renvoie theta ?

        this->maximization_step(t, y, theta, log_r, floor);
        this->log_likelihood(iteration) = this->compute_log_likelihood(log_r); // new_log_likelihood

        if (verbose >= 2)
        {
            logger.updateProgressBar(iteration);
        }
        if (verbose >= 1)
        {
            logger.log(INFO, "Iteration : " + std::to_string(iteration) + ", log likelihood : " + std::to_string(this->log_likelihood(iteration)));
        }

    } while (!this->has_converged(this->log_likelihood(iteration - 1), this->log_likelihood(iteration), iteration, max_iteration, ratio_ll, floor, verbose));
    if (verbose >= 2)
    {
        logger.stopProgressBar();
    }
    if (verbose >= 1)
    {
        logger.log(INFO, "Finish GLLiM-EM Training");
    }
    // return theta;
}

template <typename TGamma, typename TSigma>
vec EmEstimator<TGamma, TSigma>::get_log_likelihood()
{
    return this->log_likelihood;
}

// ============================== Private methods ==============================

// template <typename TGamma, typename TSigma>
// void EmEstimator<TGamma, TSigma>::expectation_W_step(const mat &t, const mat &y, GLLiMParameters<TGamma, TSigma> &theta, mat &log_r, mat &mu_w, mat &S_w)
// {
//     // TODO Clear mathematical description
//     // The posterior probability reW |Z , given parameter estimates, is fully defined by computing
//     // the distributions p(w_n |Z_n = k, t_n , y_n ; θ(i) ), for all n and all k, which can be shown to be Gaussian,
//     // with mean µ_w(n,k) and covariance matrix S_w(k) given by

//     // def _expectation_w(self, t, y, θ):
//         // if self.Lw == 0:
//         //     μw = np.zeros(0)
//         //     Sw = np.zeros(0)
//         //     return μw, Sw
//     if (this->L_w == 0)
//     {
//         mu_w.zeros();
//         S_w.zeros();
//     }

//     // if self.verbose: print('Expectation W step')
//     Logger::getInstance().log(INFO, "Finish GLLiM-EM Training");

//     // μw = np.zeros((self.Lw, self.N, self.K))
//     // Sw = np.zeros((self.Lw, self.Lw, self.K))

//     // bar = progressbar.ProgressBar(widgets=['k = ', progressbar.SimpleProgress(), progressbar.Bar()])
//     // for k in bar(range(self.K)):
//     //     if self.verbose > 1:
//     //         print('  - k = %d'%(k))

//     for (unsigned k = 0; k < theta.K; k++)
//     {
//         Logger::getInstance().log(INFO, "\t k=" << k);
//         // # DEFINITION
//         // Atk = np.reshape(θ['A'][:,0:self.Lt,k], (self.D, self.Lt), order = 'F') # DxLt
//         // Awk = np.reshape(θ['A'][:,self.Lt:self.L,k], (self.D, self.Lw), order = 'F') # DxLw
//         // bk = np.reshape(θ['b'][:,k], (self.D,1), order = 'F') # Dx1
//         // Σk = np.reshape(θ['Σ'][:,:,k], (self.D, self.D), order = 'F') # DxD
//         // Γwk = np.reshape(θ['Γ'][self.Lt:self.L, self.Lt:self.L, k], (self.Lw, self.Lw), order = 'F') # LwxLw
//         // cwk = θ['c'][self.Lt:self.L, k] # Lwx1

//         // invΓwk = np.linalg.inv(Γwk)
//         // invΣk = np.linalg.inv(Σk)
//         // invSwk = invΓwk + np.matmul(np.matmul(Awk.T, invΣk), Awk) # LwxLw

//         TGamma inv_Gamma_w_k = theta.Gamma_w[k].inv(); // (L_w, L_w)
//         TSigma inv_Sigma_k = theta.Sigma[k].inv(); // (D, D)

//         mat inv_S_w_k = inv_Gamma_w_k + theta.A_w.slice(k).t() * inv_Sigma_k * theta.A_w.slice(k) // (L_w, L_w)

//         // if not allnans(t):
//         //     Atkt = np.dot(Atk, t) # DxLt
//         // else:
//         //     Atkt = 0

//         // Sw[:,:,k] = np.linalg.inv(invSwk)
//         // μw[:,:,k] = np.dot(
//         //                 np.linalg.inv(np.dot(invSwk, Γwk)),
//         //                 np.dot(np.dot(np.dot(Γwk,Awk.T),invΣk),
//         //                 y - Atkt - bk) + cwk
//         //             )
//         // ! Fomulation différente du papier

//         S_w.slice(k) = inv_S_w_k.inv();
//         mu_w.slice(k) = S_w.slice(k) * (theta.A_w.slice(k).t() * inv_Sigma_k * ( y.col(n) - theta.A_t.slice(k) * t.col(n) - theta.B.col(k)) + inv_Gamma_w_k * theta.C_w.col(k));
//     }
//     // return μw, Sw
// }

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::expectation_Z_step(const mat &t, const mat &y, GLLiMParameters<TGamma, TSigma> &theta, mat &log_r)
{
    // TODO Hybrid

    // TODO Clear mathematical description
    // log_r(n,k) = Pi(k) p(y_n,t_n|Z_n=k;θ) / Σ(j=1:K)[Pi(j) p(y_n,t_n|Z_n=j;θ)]
    // p(y_n,t_n|Z_n=k;θ) = p(y_n|t_n,Z_n=k;θ) p(t_n|Z_n=k;θ)
    // p(y_n|t_n,Z_n=k;θ) = gaussianDensity(Y_n; A_k * X_n + B_k, Sigma_k)      (supervised formulation)
    // p(t_n|Z_n=k;θ) = gaussianDensity(X_n; C_k, Gamma_k))                     (supervised formulation)
    //  => compute log_r = log(Pi_k * gaussianDensity(Y_n; A_k * X_n + B_k, Sigma_k) * gaussianDensity(X_n; C_k, Gamma_k)) (supervised formulation)

    unsigned N = t.n_cols;
    double D_log_2_pi = theta.D * LOG_2_PI;
    double L_log_2_pi = theta.L * LOG_2_PI;

    // #pragma omp parallel for shared(N,K,L,D,x,y,theta,D_log_2_pi, L_log_2_pi,temp_density_y,temp_density_x,log_Pi_K,next_rnk)
    for (unsigned k = 0; k < theta.K; k++)
    {
        double log_det_gamma = theta.Gamma[k].log_det();
        double log_det_sigma = theta.Sigma[k].log_det();

        // compute log_r only if both the covariances have non zero determinants
        if (log_det_sigma != -datum::inf && log_det_gamma != -datum::inf)
        {
            // compute log_r only if the the weight of the k_th gaussian in the mixture is not zero
            if (theta.Pi(k) != 0)
            {
                // compute the vector (Y - A.X - B)
                mat y_u(theta.D, N);
                y_u = y - theta.A.slice(k) * t;
                y_u.each_col() -= theta.B.col(k);

                // compute the vector (X - C)
                mat x_u(theta.L, N);
                x_u = t;
                x_u.each_col() -= theta.C.col(k);

                double temp_density_y = D_log_2_pi + log_det_sigma;
                double temp_density_x = L_log_2_pi + log_det_gamma;
                // sigma_inv = inv(theta.Sigma.slice(k));
                // gamma_inv = inv(theta.Gamma.slice(k));
                TGamma gamma_inv = theta.Gamma[k].inv();
                TSigma sigma_inv = theta.Sigma[k].inv();

                double log_Pi_k = log(theta.Pi(k));

                // compute log(Pi_k * gaussianDensity(Y_n; A_k * X_n + B_k, Sigma_k) * gaussianDensity(X_n; C_k, Gamma_k))
                for (unsigned n = 0; n < N; n++)
                {
                    log_r(n, k) = log_Pi_k - 0.5 * (temp_density_x + dot(x_u.col(n), gamma_inv * vec(x_u.col(n)))) - 0.5 * (temp_density_y + dot(y_u.col(n), sigma_inv * vec(y_u.col(n))));

                    // need to test if this condition is impossible !!
                    if (log_r(n, k) == (datum::inf))
                    {
                        log_r(n, k) = -datum::inf;
                    }
                }
            }
        }
        else
        {
            // set log_r = -inf if the determinent of the covariance is equal to zero which makes the log density to tend toward +infinity
            Logger::getInstance().log(WARNING, "\tTheta Component : " + std::to_string(k) + ", Sigma log determinant : " + std::to_string(log_det_sigma) + ", Gamma log determinant : " + std::to_string(log_det_gamma));
            log_r.col(k).fill(-datum::inf);
        }
    }

    // normalization on K
    for (unsigned n = 0; n < N; n++)
    {
        log_r.row(n) -= utils::logSumExp(log_r.row(n).t());
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::maximization_step(const mat &t, const mat &y, GLLiMParameters<TGamma, TSigma> &theta, const mat &log_r, double floor) // mu, S
{
    // TODO Clear mathematical description

    unsigned N = log_r.n_rows;

    // #pragma omp parallel for shared(x, y, r_nk, next_theta, N, K, D, L) schedule(static) num_threads(2)
    for (unsigned k = 0; k < theta.K; k++)
    {
        double log_r_k = utils::logSumExp(log_r.col(k)); // r(k) = Σ(n=1:N)[r(n,k)]

        // update Pi
        this->update_Pi_k(theta, k, N, log_r_k);

        if (log_r_k != (-datum::inf)) // equivalent to (theta.Pi(k) != 0)
        {
            vec avg_r_k = exp(log_r.col(k) - log_r_k); // vector of r(n,k)/r(k)

            // update C
            this->update_C_k(theta, k, t, avg_r_k);

            // update Gamma
            this->update_Gamma_k(theta, k, t, avg_r_k);
            this->improve_covariance_stability(theta.Gamma[k], theta.L, floor);

            // Update A
            this->update_A_k(theta, k, t, y, avg_r_k);
            mat Y_AX(theta.D, N);
            Y_AX = y - theta.A.slice(k) * t;

            // update B
            this->update_B_k(theta, k, Y_AX, avg_r_k);

            // update Sigma
            this->update_Sigma_k(theta, k, Y_AX, avg_r_k);
            this->improve_covariance_stability(theta.Sigma[k], theta.D, floor);
        }
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::update_Pi_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, unsigned N, double log_r_k)
{
    theta.Pi(k) = exp(log_r_k) / N;
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::update_C_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, const mat &t, const vec &avg_r_k)
{
    theta.C.col(k).fill(0.0);
    for (unsigned n = 0; n < t.n_cols; n++)
    {
        theta.C.col(k) += t.col(n) * avg_r_k(n);
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::update_Gamma_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, const mat &t, const vec &avg_r_k)
{
    theta.Gamma[k].fill(0.0);
    for (unsigned n = 0; n < t.n_cols; n++)
    {
        // theta.Gamma[k] += avg_r_k(n) * (t.col(n) - theta.C.col(k)) * (t.col(n) - theta.C.col(k)).t();
        theta.Gamma[k].rank_one_update(t.col(n) - theta.C.col(k), avg_r_k(n)); // efficient computation
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::update_A_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, const mat &t, const mat &y, const vec &avg_r_k)
{
    // TODO Hybrid
    // TODO improve with mu and S (voir pyGLLiM)
    // NOTE: X = {T; W}
    mat x = t;

    mat X_k(x.n_rows, x.n_cols);
    mat Y_k(y.n_rows, y.n_cols);
    vec x_k(x.n_rows); // useless if x=t (supervised model/ W = 0)
    vec y_k(y.n_rows);

    for (unsigned n = 0; n < x.n_cols; n++) // useless if x=t (supervised model/ W = 0)
    {
        x_k += avg_r_k(n) * x.col(n);
    }

    for (unsigned n = 0; n < x.n_cols; n++)
    {
        y_k += avg_r_k(n) * y.col(n);
    }

    for (unsigned n = 0; n < x.n_cols; n++)
    {
        // X_k.col(n) = sqrt(avg_r_k(n)) * (x.col(n) - theta.C.col(k)); // only if x=t (supervised model/ W = 0)
        X_k.col(n) = sqrt(avg_r_k(n)) * (x.col(n) - x_k); // useless if x=t (supervised model/ W = 0)
        Y_k.col(n) = sqrt(avg_r_k(n)) * (y.col(n) - y_k);
    }

    if (accu(Y_k) != 0 && accu(X_k) != 0)
    {
        theta.A.slice(k) = Y_k * X_k.t() * pinv(X_k * X_k.t());
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::update_B_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, const mat &Y_AX, const vec &avg_r_k)
{
    theta.B.col(k).fill(0.0);
    for (unsigned n = 0; n < avg_r_k.n_rows; n++)
    {
        theta.B.col(k) += avg_r_k(n) * Y_AX.col(n);
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::update_Sigma_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, const mat &Y_AX, const vec &avg_r_k)
{
    theta.Sigma[k].fill(0.0);
    for (unsigned n = 0; n < avg_r_k.n_rows; n++)
    {
        // theta.Sigma[k] += avg_r_k(n) * (Y_AX.col(n) - theta.B.col(k)) * (Y_AX.col(n) - theta.B.col(k)).t();
        theta.Sigma[k].rank_one_update(Y_AX.col(n) - theta.B.col(k), avg_r_k(n)); // efficient computation
    }
}

template <typename TGamma, typename TSigma>
template <typename TCov>
void EmEstimator<TGamma, TSigma>::improve_covariance_stability(TCov &covariance, unsigned dimension, double floor)
{
    covariance += eye(dimension, dimension) * floor;
}

template <typename TGamma, typename TSigma>
double EmEstimator<TGamma, TSigma>::compute_log_likelihood(const mat &log_r)
{
    vec log_ll = utils::logSumExp(log_r, 0);
    return accu(log_ll) / log_r.n_cols;
}

template <typename TGamma, typename TSigma>
bool EmEstimator<TGamma, TSigma>::has_converged(double old_log_likelihood, double new_log_likelihood, unsigned current_iter, unsigned max_iteration, double ratio_ll, double floor, int verbose)
{
    double ratio_increase_likelihood = (exp(new_log_likelihood) - exp(old_log_likelihood)) / exp(old_log_likelihood);
    bool max_iter_condition = current_iter == max_iteration;

    if (max_iter_condition)
    {
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "Maximum iteration number reached");
        }
    }

    bool ratio_ll_condition = ratio_increase_likelihood <= ratio_ll / 100;
    // ratio_ll_condition = false; // TODO temporary for test. log likelihood is decreasing ?

    if (ratio_ll_condition)
    {
        if (verbose >= 1)
        {
            Logger::getInstance().log(INFO, "Likelihood increase threshold reached :" + std::to_string(ratio_ll / 100));
        }
    }
    return max_iter_condition || ratio_ll_condition;
}

// ============================== Explicit instantiation of template classes ==============================

template class EmEstimator<FullCovariance, FullCovariance>;
template class EmEstimator<FullCovariance, DiagCovariance>;
template class EmEstimator<FullCovariance, IsoCovariance>;
template class EmEstimator<DiagCovariance, FullCovariance>;
template class EmEstimator<DiagCovariance, DiagCovariance>;
template class EmEstimator<DiagCovariance, IsoCovariance>;
template class EmEstimator<IsoCovariance, FullCovariance>;
template class EmEstimator<IsoCovariance, DiagCovariance>;
template class EmEstimator<IsoCovariance, IsoCovariance>;