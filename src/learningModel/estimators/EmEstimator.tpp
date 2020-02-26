//
// Created by reverse-proxy on 19‏/2‏/2020.
//

using namespace learningModel;

template <typename T , typename U >
EmEstimator<T,U>::EmEstimator(const std::shared_ptr<EMLearningConfig> &config) {
    this->config = config;
}

template <typename T , typename U >
void EmEstimator<T,U>::estimate(const mat &x, const mat &y, std::shared_ptr<GLLiMParameters<T, U>> initial_theta) {
    mat r_nk(x.n_rows, initial_theta->Pi.n_rows, fill::zeros);

    for(unsigned iter=0; iter<(config->max_iteration); iter++){
        //std::cout << "iteration "<< iter << " : " << std::endl;
        next_rnk(x,y,initial_theta, r_nk);
        next_theta(x,y,r_nk,initial_theta);
        //initial_theta->Sigma[0].print();
    }
    std::cout << "done " << std::endl;
}

template <typename T , typename U >
void EmEstimator<T,U>::next_rnk(const mat &x, const mat &y, std::shared_ptr <GLLiMParameters<T, U>> theta, mat &next_rnk) {

    int K = theta->Pi.n_rows;
    int L = theta->C.n_rows;
    int D = theta->B.n_rows;
    int N = x.n_rows;


    for(unsigned k=0; k<K; k++){
        if(theta->Pi(k) != 0){
            double temp1 = D * log(2* datum::pi) + log(theta->Sigma[k].det());
            double temp2 = L * log(2* datum::pi) + log(theta->Gamma[k].det());
            FullCovariance sigma_inv = theta->Sigma[k].inv(false);
            FullCovariance gamma_inv = theta->Gamma[k].inv(false);

            for(unsigned n=0; n<N; n++ ){
                vec y_u = y.row(n).t() - theta->A.slice(k) * x.row(n).t() - theta->B.col(k);
                vec x_u = x.row(n).t() - theta->C.col(k);

                next_rnk(n,k) = log(theta->Pi(k)) -
                                0.5 * (temp1 +  dot((rowvec(y_u.t()) * sigma_inv).t() , y_u)) -
                                0.5 * (temp2 +  dot((rowvec(x_u.t()) * gamma_inv).t() , x_u));

                if(next_rnk(n,k) == (datum::inf)){
                    next_rnk(n,k) = -datum::inf;
                }
            }
        }
    }


    /*double log_l = 0;
    for(unsigned n=0; n<N; n++ ){
        double result = 0;
        for(unsigned k=0; k<K; k++){
            result += exp(next_rnk(n,k) - next_rnk.row(n).max());
        }
        log_l += (log(result) + next_rnk.row(n).max());
    }

    std::cout << "log_vraissamblance " << log_l/N << std::endl;*/

    double sum = 0;
    for(unsigned n=0; n<N; n++ ){
        sum = 0;
        if(next_rnk.row(n).max() != (-datum::inf)){
            for(unsigned k=0; k<K; k++){
                sum += exp(next_rnk(n,k) - next_rnk.row(n).max());
            }
            next_rnk.row(n) -= (log(sum) + next_rnk.row(n).max());
        }
    }

}

template <typename T , typename U >
void EmEstimator<T,U>::next_theta(const mat &x, const mat &y, const mat &r_nk,
                             std::shared_ptr <GLLiMParameters<T, U>> &next_theta) {

    int N = r_nk.n_rows;
    int K = r_nk.n_cols;
    int L = x.n_cols;
    int D = y.n_cols;
    double r_k = 0;

    for(unsigned k=0; k<K; k++){

        r_k = 0;
        double sum = 0;

        for(unsigned n=0; n<N; n++ ){
            sum += exp(r_nk(n,k) - r_nk.col(k).max());
        }
        r_k = (log(sum) + r_nk.col(k).max());

        // Update Pi
        next_theta->Pi(k) = exp(r_k)/N;


        if(exp(r_k) != 0){
            // Update C
            next_theta->C.col(k).fill(0.0);
            for(unsigned n=0; n<N; n++) {
                next_theta->C.col(k) = next_theta->C.col(k) + vec(x.row(n).t()) * exp(r_nk(n,k) - r_k);
            }

            // Update Gamma
            next_theta->Gamma[k] = 0.0;
            for(unsigned n=0; n<N; n++){
                mat temp = mat(x.row(n).t() - next_theta->C.col(k));
                next_theta->Gamma[k] = next_theta->Gamma[k] + temp * temp.t() * exp(r_nk(n,k) - r_k);
            }


            // Update A
            mat X_k(L,N);
            mat Y_k(D,N);
            rowvec x_k = rowvec(next_theta->C.col(k).t()); //same formula
            rowvec y_k(D, fill::zeros);
            for(unsigned n=0; n<N; n++){
                y_k += y.row(n) * exp(r_nk(n,k) - r_k);
            }

            for(unsigned n=0; n<N; n++){
                X_k.col(n) = exp(0.5 * (r_nk(n,k) - r_k)) * vec((x.row(n)- x_k).t());
                Y_k.col(n) = exp(0.5 * (r_nk(n,k) - r_k)) * vec((y.row(n)- y_k).t());
            }

            next_theta->A.slice(k) = mat(Y_k * X_k.t()) * pinv(X_k * X_k.t());

            //update B
            next_theta->B.col(k).fill(0.0);
            for(unsigned n=0; n<N; n++){
                next_theta->B.col(k) = next_theta->B.col(k) + vec(y.row(n).t() - next_theta->A.slice(k) * x.row(n).t()) * exp(r_nk(n,k) - r_k);
            }

            // Update Sigma
            next_theta->Sigma[k] = 0.0;
            for(unsigned n=0; n<N; n++){
                next_theta->Sigma[k] = next_theta->Sigma[k] + mat(y.row(n).t() - next_theta->A.slice(k) * x.row(n).t() - next_theta->B.col(k))
                                                              * mat(y.row(n).t() - next_theta->A.slice(k) * x.row(n).t() - next_theta->B.col(k)).t() *
                                                              exp(r_nk(n,k) - r_k);
            }
        }
    }

}