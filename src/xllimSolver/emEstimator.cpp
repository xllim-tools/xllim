#include "omp.h"
#include "emEstimator.hpp"
#include "../utils/utils.hpp"
#include "../logging/logger.hpp"

// ! Useful notation
// A_w_k = theta.A.slice(k).tail_cols(theta.L_w) // TODO is it possible to do an alias ?

#define LOG_2_PI log(2 * datum::pi)

template <typename TGamma, typename TSigma>
EmEstimator<TGamma, TSigma>::EmEstimator() {}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::train(const mat &t, const mat &y, GLLiMParameters<TGamma, TSigma> &theta, unsigned max_iteration, double ratio_ll, double floor, int verbose)
{
    mat log_r(t.n_cols, theta.Pi.n_cols, fill::value(-datum::inf)); // Posterior log probability (N, K)
    cube mu_w(theta.L_w, t.n_cols, theta.Pi.n_cols);                // Gaussian mean of posterior probability r_W|Z (L_w, N, K)
    cube S_w(theta.L_w, theta.L_w, theta.Pi.n_cols);                // Gaussian covariance matrix of posterior probability r_W|Z (L_w, L_w, K)

    log_likelihood_ = vec(max_iteration + 1, fill::value(-datum::inf)); // A list of log-likelihood at each iteration

    // set a logger with a progress bar for GLLiM-EM training
    Logger &logger = Logger::getInstance();
    if (verbose >= 2)
    {
        logger.startProgressBar(max_iteration);
    }
    logger.log(INFO, 1, verbose, "[Training] Start GLLiM-EM");

    unsigned iteration = 0;
    do
    {
        iteration++;
        expectation_W_step(t, y, theta, mu_w, S_w);
        expectation_Z_step(t, y, theta, log_r, iteration);
        maximization_step(t, y, theta, log_r, mu_w, S_w, floor);

        if (verbose >= 2)
        {
            logger.updateProgressBar(iteration);
        }
        logger.log(INFO, 1, verbose, "\tIteration : " + std::to_string(iteration) + ", avg log-likelihood : " + std::to_string(log_likelihood_(iteration)));

    } while (!has_converged(log_likelihood_(iteration - 1), log_likelihood_(iteration), iteration, max_iteration, ratio_ll, floor, verbose));

    if (verbose >= 2)
    {
        logger.stopProgressBar();
    }
    logger.log(INFO, 1, verbose, "[Training] GLLiM-EM completed");
}

template <typename TGamma, typename TSigma>
vec EmEstimator<TGamma, TSigma>::get_log_likelihood()
{
    return log_likelihood_;
}

// ============================== Private methods ==============================

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::expectation_W_step(const mat &t, const mat &y, GLLiMParameters<TGamma, TSigma> &theta, cube &mu_w, cube &S_w)
{
    // The posterior probability r (W |Z) , given parameter estimates, is fully defined by computing
    // the distributions p(w_n |Z_n = k, t_n , y_n ; θ(i) ), for all n and all k, which can be shown to be Gaussian,
    // with mean µ_w(n,k) and covariance matrix S_w(k) given by (see paper)

    if (theta.L_w > 0)
    {
        for (unsigned k = 0; k < theta.K; k++)
        {
            TGamma inv_Gamma_w_k = theta.Gamma[k].tail(theta.L_w).inv(); // (L_w, L_w)
            TSigma inv_Sigma_k = theta.Sigma[k].inv();                   // (D, D)

            // compute S_w
            mat inv_S_w_k = inv_Gamma_w_k.get_mat() + mat(theta.A.slice(k).tail_cols(theta.L_w).t()) * inv_Sigma_k * theta.A.slice(k).tail_cols(theta.L_w); // (L_w, L_w)
            S_w.slice(k) = inv(inv_S_w_k);                                                                                                                  // Armadillo inv() method

            // compute mu_w
            // ! Fomulation papier et pyGLLiM différente
            // Sw[:,:,k] = np.linalg.inv(invSwk)
            // μw[:,:,k] = np.dot(  np.linalg.inv( np.dot(invSwk, Γwk)), np.dot( np.dot( np.dot(Γwk,Awk.T), invΣk),   y - Atkt - bk) + cwk )
            for (size_t n = 0; n < mu_w.n_cols; n++)
            {
                mu_w.slice(k).col(n) = S_w.slice(k) * (mat(theta.A.slice(k).tail_cols(theta.L_w).t()) * inv_Sigma_k * (y.col(n) - theta.A.slice(k).head_cols(theta.L_t) * t.col(n) - theta.B.col(k)) + inv_Gamma_w_k * vec(theta.C.col(k).tail(theta.L_w)));
            }
        }
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::expectation_Z_step(const mat &t, const mat &y, GLLiMParameters<TGamma, TSigma> &theta, mat &log_r, unsigned iteration)
{
    // The posterior probability r (r_Z in the paper) is defined by:
    // r(n,k)               = Pi(k) p(y_n,t_n|Z_n=k;θ) / Σ(j=1:K)[Pi(j) p(y_n,t_n|Z_n=j;θ)]
    // p(y_n,t_n|Z_n=k;θ)   = p(y_n|t_n,Z_n=k;θ) p(t_n|Z_n=k;θ)
    // p(y_n|t_n,Z_n=k;θ)   = gaussianDensity(y_n; A_k * [t_n;C_w_k] + B_k, Sigma_k + A_w_k * Gamma_w_k * A_w_k^T)  = "density_t"
    // p(t_n|Z_n=k;θ)       = gaussianDensity(t_n; C_t_k, Gamma_t_k))                                               = "density_y"
    // => Finally     log_r = log(Pi_k) + log(density_t) + log(density_y)
    // => Normalization on axis K

    unsigned N = t.n_cols;
    double D_log_2_pi = theta.D * LOG_2_PI;
    double L_log_2_pi = theta.L * LOG_2_PI;

    // #pragma omp parallel for shared(N,K,L,D,x,y,theta,D_log_2_pi, L_log_2_pi,temp_density_y,temp_density_x,log_Pi_K,next_rnk)
    for (unsigned k = 0; k < theta.K; k++)
    {
        // compute log_r only if the the weight of the k_th gaussian in the mixture is not zero
        if (theta.Pi(k) != 0)
        {
            double log_det_Gamma_t_k = theta.Gamma[k].head(theta.L_t).log_det();
            double log_det_Sigma_k;
            if (theta.L_w > 0)
            {
                log_det_Sigma_k = log_det(theta.A.slice(k).tail_cols(theta.L_w) * theta.Gamma[k].tail(theta.L_w) * theta.A.slice(k).tail_cols(theta.L_w).t() + theta.Sigma[k]).real(); // Armadillo log_det() method
            }
            else
            {
                log_det_Sigma_k = theta.Sigma[k].log_det(); // xCovariance log_det() method
            }

            // compute log_r only if both the covariances have non zero determinants
            if (log_det_Sigma_k != -datum::inf && log_det_Gamma_t_k != -datum::inf)
            {
                double log_Pi_k = log(theta.Pi(k));
                double const_density_t = L_log_2_pi + log_det_Gamma_t_k;
                double const_density_y = D_log_2_pi + log_det_Sigma_k;

                TGamma Gamma_t_k_inv = theta.Gamma[k].head(theta.L_t).inv();
                TSigma Sigma_k_inv(theta.D);
                if (theta.L_w > 0)
                {
                    Sigma_k_inv = TSigma(mat(inv(theta.A.slice(k).tail_cols(theta.L_w) * theta.Gamma[k].tail(theta.L_w) * theta.A.slice(k).tail_cols(theta.L_w).t() + theta.Sigma[k]))); // Armadillo inv() method //! if TSigma is Diag or Iso and L_w > 0, Sigma_k_inv should be of type mat !
                }
                else
                {
                    Sigma_k_inv = theta.Sigma[k].inv(); // xCovariance inv() method
                }

                // compute all vector (y_n - A_k*[t_n;C_w_k] - B_k)
                mat y_u = y - theta.A.slice(k).head_cols(theta.L_t) * t;
                y_u.each_col() -= theta.B.col(k);
                if (theta.L_w > 0)
                {
                    y_u.each_col() -= theta.A.slice(k).tail_cols(theta.L_w) * theta.C.col(k).tail(theta.L_w);
                }

                // compute all vector (t_n - C_t_k)
                mat t_u = t;
                t_u.each_col() -= theta.C.col(k).head(theta.L_t);

                for (unsigned n = 0; n < N; n++)
                {
                    log_r(n, k) = log_Pi_k - 0.5 * (const_density_t + dot(t_u.col(n), Gamma_t_k_inv * vec(t_u.col(n)))) - 0.5 * (const_density_y + dot(y_u.col(n), Sigma_k_inv * vec(y_u.col(n)))); // log_r = log(Pi_k) + log(density_t) + log(density_y)
                }
            }
            else
            {
                // set log_r = -inf if the determinent of the covariance is equal to zero which makes the log density to tend toward +infinity
                Logger::getInstance().log(WARNING, "\tTheta Component : " + std::to_string(k) + ", Sigma log determinant : " + std::to_string(log_det_Sigma_k) + ", Gamma log determinant : " + std::to_string(log_det_Gamma_t_k));
                log_r.col(k).fill(-datum::inf);
            }
        }
    }

    // Vector of shape (N) corresponding to Σ(j=1:K)[Pi(j) p(y_n,t_n|Z_n=j;θ)]
    // It is useful for log_r normalisation and computing average log-likelihood
    vec log_r_n = utils::logSumExp(log_r, 1);

    // compute average log-likelihood
    //      The observed-data log-likelihood is defined by:
    //      L(t,θ)   = Σ(n=1:N)[ log_p(y_n,t_n;θ) ]
    //               = Σ(n=1:N)[ log( Σ(k=1:K)[ Pi(k) p(y_n,t_n|Z_n=k;θ) ] ) ]
    //               = Σ(n=1:N)[ Σ(k=1:K)[ Pi(k) p(y_n,t_n|Z_n=k;θ) ] ]
    log_likelihood_(iteration) = accu(log_r_n) / N;

    // normalization on K
    log_r.each_col() -= log_r_n;
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::maximization_step(const mat &t, const mat &y, GLLiMParameters<TGamma, TSigma> &theta, const mat &log_r, const cube &mu_w, const cube &S_w, double floor)
{
    // TODO Clear mathematical description

    unsigned N = log_r.n_rows;

    // #pragma omp parallel for shared(x, y, r_nk, next_theta, N, K, D, L) schedule(static) num_threads(2)
    for (unsigned k = 0; k < theta.K; k++)
    {
        double log_r_k = utils::logSumExp(log_r.col(k)); // r(k) = Σ(n=1:N)[r(n,k)]

        // update Pi
        update_Pi_k(theta, k, N, log_r_k);

        if (log_r_k != (-datum::inf)) // equivalent to (theta.Pi(k) != 0)
        {
            vec avg_r_k = exp(log_r.col(k) - log_r_k); // vector of r(n,k)/r(k)

            // update C
            update_C_k(theta, k, t, avg_r_k);

            // update Gamma
            update_Gamma_k(theta, k, t, avg_r_k);
            improve_covariance_stability(theta.Gamma[k], theta.L, floor);

            // Update A
            update_A_k(theta, k, t, y, avg_r_k, mu_w.slice(k), S_w.slice(k));
            mat Y_AX(theta.D, N);
            mat x = join_cols(t, mu_w.slice(k));
            Y_AX = y - theta.A.slice(k) * x;

            // update B
            update_B_k(theta, k, Y_AX, avg_r_k);

            // update Sigma
            update_Sigma_k(theta, k, Y_AX, avg_r_k, S_w.slice(k));
            improve_covariance_stability(theta.Sigma[k], theta.D, floor);
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
    theta.C.col(k).head(theta.L_t).fill(0.0);
    for (unsigned n = 0; n < t.n_cols; n++)
    {
        theta.C.col(k).head(theta.L_t) += t.col(n) * avg_r_k(n);
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::update_Gamma_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, const mat &t, const vec &avg_r_k)
{
    theta.Gamma[k].fill_head(theta.L_t, 0.0);
    for (unsigned n = 0; n < t.n_cols; n++)
    {
        theta.Gamma[k].rank_one_update_head(theta.L_t, t.col(n) - theta.C.col(k).head(theta.L_t), avg_r_k(n)); // efficient computation
    }
}

template <typename TGamma, typename TSigma>
void EmEstimator<TGamma, TSigma>::update_A_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, const mat &t, const mat &y, const vec &avg_r_k, const mat &mu_w_k, const mat &S_w_k)
{
    mat x_k = join_cols(t, mu_w_k); // X = {T; W} and then the estimated x(k) is {t; mu_w(k)}  (L,N)

    mat X_k(x_k.n_rows, x_k.n_cols); // (L,N)
    mat Y_k(y.n_rows, y.n_cols);     // (D,N)
    vec x_k_mean(x_k.n_rows);        // (L)
    vec y_k_mean(y.n_rows);          // (D)

    // compute x_k weighted mean
    x_k_mean.head(theta.L_t) = theta.C.col(k).head(theta.L_t); // This sum has already been computed for the observed part (L_t first elements)
    if (theta.L_w > 0)
    {
        for (unsigned n = 0; n < x_k.n_cols; n++)
        {
            x_k_mean.tail(theta.L_w) += avg_r_k(n) * mu_w_k.col(n);
        }
    }

    // compute y_k weighted mean
    for (unsigned n = 0; n < x_k.n_cols; n++)
    {
        y_k_mean += avg_r_k(n) * y.col(n);
    }

    // build X_k and Y_k matrices
    for (unsigned n = 0; n < x_k.n_cols; n++)
    {
        X_k.col(n) = sqrt(avg_r_k(n)) * (x_k.col(n) - x_k_mean);
        Y_k.col(n) = sqrt(avg_r_k(n)) * (y.col(n) - y_k_mean);
    }

    // construct A_k
    mat X_k_quadratic = X_k * X_k.t(); // (L,L)
    if (theta.L_w > 0)
    {
        X_k_quadratic.submat(theta.L_t, theta.L_t, theta.L - 1, theta.L - 1) += S_w_k;
    }
    if (accu(Y_k) != 0 && accu(X_k) != 0)
    {
        theta.A.slice(k) = Y_k * X_k.t() * pinv(X_k_quadratic); // TODO it should be inv_sympd
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
void EmEstimator<TGamma, TSigma>::update_Sigma_k(GLLiMParameters<TGamma, TSigma> &theta, unsigned k, const mat &Y_AX, const vec &avg_r_k, const mat &S_w_k)
{
    theta.Sigma[k] = mat(theta.A.slice(k).tail_cols(theta.L_w) * S_w_k * theta.A.slice(k).tail_cols(theta.L_w).t());
    for (unsigned n = 0; n < avg_r_k.n_rows; n++)
    {
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
bool EmEstimator<TGamma, TSigma>::has_converged(double old_log_likelihood, double new_log_likelihood, unsigned current_iter, unsigned max_iteration, double ratio_ll, double floor, int verbose)
{
    double ratio_increase_likelihood = (exp(new_log_likelihood) - exp(old_log_likelihood)) / exp(old_log_likelihood);
    bool max_iter_condition = current_iter == max_iteration;

    if (max_iter_condition)
    {
        Logger::getInstance().log(WARNING, 1, verbose, "[Training] Maximum iteration number reached");
    }

    bool ratio_ll_condition = ratio_increase_likelihood <= ratio_ll / 100;
    // ratio_ll_condition = false; // TODO temporary for test. log likelihood is decreasing ?

    if (ratio_ll_condition)
    {
        log_likelihood_ = vec(log_likelihood_.head(current_iter + 1)); // reduce log_likelihood_ vec size
        Logger::getInstance().log(WARNING, 1, verbose, "[Training] Likelihood increase threshold reached :" + std::to_string(ratio_ll / 100));
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