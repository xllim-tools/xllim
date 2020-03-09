//
// Created by reverse-proxy on 19‏/2‏/2020.
//

#include "omp.h"


#define LOG_2_PI log(2* datum::pi)

using namespace learningModel;

template <typename T , typename U >
EmEstimator<T,U>::EmEstimator(const std::shared_ptr<EMLearningConfig> &config) {
    this->config = config;
}

template<typename T, typename U>
EmEstimator<T, U>::EmEstimator() {
    this->config = std::make_shared<EMLearningConfig>();
}

template <typename T , typename U >
void EmEstimator<T,U>::execute(const mat &x, const mat &y, std::shared_ptr<GLLiMParameters<T, U>> initial_theta) {
    mat r_nk(x.n_rows, initial_theta->Pi.n_rows, fill::zeros);

    mat x_t = x.t();
    mat y_t = y.t();

    auto start1 = std::chrono::high_resolution_clock::now();
    auto start2 = std::chrono::high_resolution_clock::now();
    auto end1 = std::chrono::high_resolution_clock::now();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);


    for(unsigned iter=0; iter<(config->max_iteration); iter++){
        std::cout << "iteration "<< iter << " : " << std::endl;

        start1 = std::chrono::high_resolution_clock::now();
        next_rnk(x_t,y_t,initial_theta, r_nk);
        end1 = std::chrono::high_resolution_clock::now();
        duration1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

        std::cout << "log_likelihood " << log_likelihood(r_nk) << std::endl;

        start2 = std::chrono::high_resolution_clock::now();
        next_theta(x_t,y_t,r_nk,initial_theta);
        end2 = std::chrono::high_resolution_clock::now();
        duration2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    }

    cout << duration1.count() << endl;
    cout << duration2.count() << endl;
    std::cout << "done " << std::endl;
}

template <typename T , typename U >
void EmEstimator<T,U>::next_rnk(const mat &x, const mat &y, std::shared_ptr <GLLiMParameters<T, U>> theta, mat &next_rnk) {

    int K = theta->Pi.n_rows;
    int L = theta->C.n_rows;
    int D = theta->B.n_rows;
    int N = x.n_cols;

    double D_log_2_pi = D * LOG_2_PI;
    double L_log_2_pi = L * LOG_2_PI;
    double temp_density_y = 0;
    double temp_density_x = 0;
    double log_Pi_K = 0;
    double det_sigma;
    double det_gamma;

    U sigma_inv;
    T gamma_inv;
    mat y_u(D, N, fill::zeros);
    mat x_u(L, N, fill::zeros);


//#pragma omp parallel for shared(N,K,L,D,x,y,theta,D_log_2_pi, L_log_2_pi,temp_density_y,temp_density_x,log_Pi_K,next_rnk)
    for(unsigned k=0; k<K; k++){
        det_sigma = theta->Sigma[k].det();
        det_gamma = theta->Gamma[k].det();

        // compute rnk only if both the covariances have non zero determinants
        if(det_sigma != 0 && det_gamma != 0){
            // compute rnk only if the the weight of the k_th gaussian in the mixture is not zero
            if(theta->Pi(k) != 0){
                // compute the vector (Y - A.X - B)
                y_u = y - theta->A.slice(k) * x;
                y_u.each_col() -= theta->B.col(k);

                // compute the vector (X - C)
                x_u = x;
                x_u.each_col() -= theta->C.col(k);

                temp_density_y = D_log_2_pi + log(det_sigma);
                temp_density_x = L_log_2_pi + log(det_gamma);
                sigma_inv = theta->Sigma[k].inv();
                gamma_inv = theta->Gamma[k].inv();

                log_Pi_K = log(theta->Pi(k));

                // compute log(Pi_k * gaussianDensity(Y_n; A_k * X_n + B_k, Sigma_k) * gaussianDensity(X_n; C_k, Gamma_k))
                for(unsigned n=0; n<N; n++ ){
                    next_rnk(n,k) = log_Pi_K -
                                    0.5 * (temp_density_y +  dot((rowvec(y_u.col(n).t()) * sigma_inv).t() , y_u.col(n))) -
                                    0.5 * (temp_density_x +  dot((rowvec(x_u.col(n).t()) * gamma_inv).t() , x_u.col(n)));

                    //need to test if this condition is impossible !!
                    if(next_rnk(n,k) == (datum::inf)){
                        next_rnk(n,k) = -datum::inf;
                    }
                }
            }
        }
        else{
            // set rnk = -inf if the determinent of the covariance is equal to zero which makes the log density
            // to tend toward +infinity
            next_rnk.col(k).fill(-datum::inf);
        }
    }
}

template <typename T , typename U >
void EmEstimator<T,U>::next_theta(const mat &x, const mat &y, const mat &r_nk,
                             std::shared_ptr <GLLiMParameters<T, U>> next_theta) {

    int N = r_nk.n_rows;
    int K = r_nk.n_cols;
    int L = x.n_rows;
    int D = y.n_rows;
    mat Y_AX(D,N);
    vec exp_avg_rnk(N);
    double r_k = 0;

    // normalize log_rnk
    mat log_rnk_norm = norm_log_rnk(r_nk);

//#pragma omp parallel for shared(x, y, r_nk, next_theta, N, K, D, L) schedule(static) num_threads(2)
    for(unsigned k=0; k<K; k++){
        r_k = logSumExp(log_rnk_norm.col(k));
        if(r_k != (-datum::inf)){
            exp_avg_rnk = exp(log_rnk_norm.col(k) - r_k);
        }

        // Update Pi
        update_Pi_k(next_theta, k, N, r_k);

        if(next_theta->Pi(k) != 0){
            // Update C
            update_C_k(next_theta,k,x,exp_avg_rnk);

            // Update Gamma
            update_Gamma_k(next_theta,k,x,exp_avg_rnk);
            covStabilityImprov(next_theta->Gamma[k], L, config->floor);

            // Update A
            update_A_k(next_theta,k,x,y,exp_avg_rnk);
            Y_AX = y - next_theta->A.slice(k) * x;

            //update B
            update_B_k(next_theta, k, Y_AX, exp_avg_rnk);

            //update Sigma
            update_Sigma_k(next_theta, k, Y_AX, exp_avg_rnk);
            covStabilityImprov(next_theta->Sigma[k], D, config->floor);
        }
    }
}

template <typename T , typename U>
double EmEstimator<T,U>::logSumExp(const vec &elements) {
    double result = 0;
    double max = elements.max();

    if(max == -datum::inf){
        return max;
    }else{
        for(unsigned i=0; i<elements.n_rows; i++){
            result += exp(elements(i) - max);
        }
        result = log(result) + max;
        return result;
    }
}

template <typename T , typename U>
void EmEstimator<T,U>::update_Pi_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, unsigned N, double r_k) {
    next_theta->Pi(k) = exp(r_k)/N;
}

template <typename T , typename U>
void EmEstimator<T,U>::update_A_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const mat &y, const vec &exp_avg_rnk) {
    mat X_k(x.n_rows, x.n_cols);
    mat Y_k(y.n_rows, y.n_cols);
    vec y_k(y.n_rows);

    y_k.fill(0);
    for(unsigned n=0; n<x.n_cols; n++){
        y_k = y_k + y.col(n) * exp_avg_rnk(n);
    }

    for(unsigned n=0; n<x.n_cols; n++){
        X_k.col(n) = sqrt(exp_avg_rnk(n)) * (x.col(n)- next_theta->C.col(k));
        Y_k.col(n) = sqrt(exp_avg_rnk(n)) * (y.col(n)- y_k);
    }

    if( accu(Y_k) != 0 && accu(X_k) != 0){
        next_theta->A.slice(k) = Y_k * X_k.t() * pinv(X_k * X_k.t());
    }
}

template <typename T , typename U>
void EmEstimator<T,U>::update_B_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &Y_AX, const vec &exp_avg_rnk) {
    next_theta->B.col(k).fill(0.0);
    for(unsigned n=0; n<exp_avg_rnk.n_rows ; n++){
        next_theta->B.col(k) += Y_AX.col(n) * exp_avg_rnk(n);
    }
}

template <typename T , typename U>
void EmEstimator<T,U>::update_C_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const vec &exp_avg_rnk) {
    next_theta->C.col(k).fill(0.0);
    for(unsigned n=0; n<x.n_cols; n++) {
        next_theta->C.col(k) += x.col(n) * exp_avg_rnk(n);
    }
}

template <typename T , typename U>
void EmEstimator<T,U>::update_Sigma_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &Y_AX, const vec &exp_avg_rnk) {
    next_theta->Sigma[k] = 0.0;
    for(unsigned n=0; n<exp_avg_rnk.n_rows ; n++){
        next_theta->Sigma[k].rankOneUpdate(Y_AX.col(n) - next_theta->B.col(k), exp_avg_rnk(n));
    }
}

template <typename T , typename U>
void EmEstimator<T,U>::update_Gamma_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const vec &exp_avg_rnk) {
    next_theta->Gamma[k] = 0.0;
    for(unsigned n=0; n<x.n_cols; n++){
        next_theta->Gamma[k].rankOneUpdate(x.col(n) - next_theta->C.col(k), exp_avg_rnk(n));
    }
}

template<typename T, typename U>
template<typename V>
void EmEstimator<T, U>::covStabilityImprov(V &covariance, unsigned dimension, double floor) {
    static_assert(std::is_base_of<Icovariance, V>(), "Type V must be Icovariance specialization");
    covariance += eye(dimension ,dimension ) * floor;
}

template<typename T, typename U>
mat EmEstimator<T, U>::norm_log_rnk(const mat &r_nk) {
    mat norl_log_rnk = r_nk;
    double sum = 0;
    for(unsigned n=0; n<r_nk.n_rows ; n++ ){
        sum = logSumExp(norl_log_rnk.row(n).t());
        if(sum != (-datum::inf)){
            norl_log_rnk.row(n) -= sum;
        }
    }
    return norl_log_rnk;
}

template<typename T, typename U>
double EmEstimator<T, U>::log_likelihood(const mat& r_nk) {
    double log_l = 0;
    for(unsigned n=0; n<r_nk.n_rows; n++ ){
        log_l += logSumExp(r_nk.row(n).t());
    }
    return log_l/r_nk.n_rows;
}





