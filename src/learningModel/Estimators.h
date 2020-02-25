//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_ESTIMATORS_H
#define KERNELO_ESTIMATORS_H

#include <armadillo>
#include "Icovariance.h"
#include "GLLiMParameters.h"
#include "LearningConfig.h"
#include <memory>

namespace learningModel{

    template <typename T , typename U >
    class Iestimator{

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        virtual void estimate(const mat& x, const mat& y, std::shared_ptr<GLLiMParameters<T,U>> initial_theta) = 0;

    };

    template <typename T , typename U >
    class EmEstimator : public Iestimator<T,U>{

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        explicit EmEstimator(const std::shared_ptr<EMLearningConfig>& config){
            this->config = config;
        };

        void next_rnk(const mat& x, const mat& y, std::shared_ptr<GLLiMParameters<T, U>> theta, mat &next_rnk){
            int K = theta->Pi.n_rows;
            int L = theta->C.n_rows;
            int D = theta->B.n_rows;
            int N = x.n_rows;

            //next_rnk.row(0).print();

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

                        if(n==0){
                            //(rowvec(y_u.t()) * sigma_inv).t().print("prod");
                            //std::cout << "nan1 : " << theta->Sigma[k].det() << std::endl;
                            //std::cout << "nan2 : " << theta->Gamma[k].det() << std::endl;
                            //std::cout << "nan3 : " << dot((rowvec(y_u.t()) * sigma_inv).t() , y_u) << std::endl;
                            //std::cout << "nan4 : " << dot((x_u * gamma_inv).t() , x_u) << std::endl;
                        }

                    }
                }
            }

            //std::cout << "rnk00 : " << next_rnk(0,0) << std::endl;


            double log_l = 0;
            for(unsigned n=0; n<N; n++ ){
                double result = 0;
                for(unsigned k=0; k<K; k++){
                    result += exp(next_rnk(n,k) - next_rnk.row(n).max());
                }
                log_l += (log(result) + next_rnk.row(n).max());
            }

            std::cout << "log_vraissamblance " << log_l/N << std::endl;

            double sum = 0;
            for(unsigned n=0; n<N; n++ ){
                sum = 0;
                if(next_rnk.row(n).max() != (-datum::inf)){
                    for(unsigned k=0; k<K; k++){
                        sum += exp(next_rnk(n,k) - next_rnk.row(n).max());
                    }
                    next_rnk.row(n) -= (log(sum) + next_rnk.row(n).max());
                    //std::cout << "sum_rnk" << accu(exp(next_rnk.row(n))) << std::endl;
                }
            }
            //next_rnk.row(0).print();
        }

        void next_theta(const mat& x, const mat& y, const mat& r_nk, std::shared_ptr<GLLiMParameters<T, U>> &next_theta){
            int N = r_nk.n_rows;
            int K = r_nk.n_cols;
            int L = x.n_cols;
            int D = y.n_cols;
            double r_k = 0;

            //r_nk.row(0).print();


            for(unsigned k=0; k<K; k++){
                //std::cout << "K : " << k <<std::endl;
                r_k = 0;
                double sum = 0;

                for(unsigned n=0; n<N; n++ ){
                    sum += exp(r_nk(n,k) - r_nk.col(k).max());
                }
                r_k = (log(sum) + r_nk.col(k).max());

                //std::cout << "r_k : "<< r_k << std::endl;


                // Update Pi
                //std::cout << "K : "<< k << "Pi" << std::endl;
                next_theta->Pi(k) = exp(r_k)/N;
                //next_theta->Pi.print("Pi");

                if(exp(r_k) != 0){
                    // Update C
                    //std::cout << "K : "<< k << "C" << std::endl;
                    next_theta->C.col(k).fill(0.0);
                    for(unsigned n=0; n<N; n++) {
                        next_theta->C.col(k) = next_theta->C.col(k) + vec(x.row(n).t()) * exp(r_nk(n,k) - r_k);
                    }
                    //next_theta->C.col(k).t().print("C");



                    // Update Gamma
                    //std::cout << "K : "<< k << "Gamma" << std::endl;
                    next_theta->Gamma[k] = 0.0;
                    for(unsigned n=0; n<N; n++){
                        mat temp = mat(x.row(n).t() - next_theta->C.col(k));
                        next_theta->Gamma[k] = next_theta->Gamma[k] + temp * temp.t() * exp(r_nk(n,k) - r_k);
                    }
                    /*std::cout << "Gamma : " << std::endl;
                    next_theta->Gamma[k].print();*/

                    // Update A

                    mat X_k(L,N);
                    mat Y_k(D,N);
                    rowvec x_k = rowvec(next_theta->C.col(k).t()); //same formula
                    rowvec y_k(D, fill::zeros);
                    for(unsigned n=0; n<N; n++){
                        y_k += y.row(n) * exp(r_nk(n,k) - r_k);
                    }



                    //(x.row(0)- x_k).print();
                    //std::cout << "X_k.has_nan() "<< x(0,)) << std::endl;

                    for(unsigned n=0; n<N; n++){
                        X_k.col(n) = exp(0.5 * (r_nk(n,k) - r_k)) * vec((x.row(n)- x_k).t());
                        Y_k.col(n) = exp(0.5 * (r_nk(n,k) - r_k)) * vec((y.row(n)- y_k).t());
                    }
                    //std::cout << "K : "<< k << "A" << std::endl;

                    /*x.print("X");
                    y.print("Y");
                    x_k.print("x_k");
                    y_k.print("y_k");
                    X_k.print("X_k");
                    Y_k.print("Y_k");*/
                    //r_nk.print("rnk");

                    next_theta->A.slice(k) = mat(Y_k * X_k.t()) * pinv(X_k * X_k.t());

                    //next_theta->A.slice(k).print("A");


                    //update B
                    //std::cout << "K : "<< k << "B" << std::endl;
                    next_theta->B.col(k).fill(0.0);
                    //next_theta->B.col(k).t().print("B0");
                    for(unsigned n=0; n<N; n++){
                        next_theta->B.col(k) = next_theta->B.col(k) + vec(y.row(n).t() - next_theta->A.slice(k) * x.row(n).t()) * exp(r_nk(n,k) - r_k);
                        /*std::cout << "exp(r_nk(n,k) - r_k)" << exp(r_nk(n,k) - r_k) << std::endl;
                        vec(y.row(n).t() - next_theta->A.slice(k) * x.row(n).t()).print("Y - AX");
                        next_theta->B.col(k).t().print("B");*/
                    }



                    // Update Sigma
                    //std::cout << "K : "<< k << "Sigma" << std::endl;
                    next_theta->Sigma[k] = 0.0;


                    for(unsigned n=0; n<N; n++){
                        next_theta->Sigma[k] = next_theta->Sigma[k] + mat(y.row(n).t() - next_theta->A.slice(k) * x.row(n).t() - next_theta->B.col(k))
                                               * mat(y.row(n).t() - next_theta->A.slice(k) * x.row(n).t() - next_theta->B.col(k)).t() *
                                               exp(r_nk(n,k) - r_k);
                    }
                    //std::cout << "prev Sigma : " << std::endl;
                    //next_theta->Sigma[k].print();

                }


            }
        }

        void estimate(
                const mat& x,
                const mat& y,
                std::shared_ptr<GLLiMParameters<T, U>> initial_theta) override{

            mat r_nk(x.n_rows, initial_theta->Pi.n_rows, fill::zeros);

            //r_nk.row(0).print();

            for(unsigned iter=0; iter<(config->max_iteration); iter++){
                std::cout << "iteration "<< iter << " : " << std::endl;
                next_rnk(x,y,initial_theta, r_nk);
                next_theta(x,y,r_nk,initial_theta);
                //initial_theta->Sigma[0].print();
            }
            std::cout << "done " << std::endl;
        };

    private:
        std::shared_ptr<EMLearningConfig> config;
    };


    class GmmEstimator: public Iestimator<FullCovariance, FullCovariance>{

    public:
        explicit GmmEstimator(const std::shared_ptr<GMMLearningConfig>& config){
            this->config = config;
        };
        GmmEstimator() = default;

        void estimate(
                const mat& x,
                const mat& y,
                std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> initial_theta)  {
            // transform GLLiM parameters to GMM parameters
            this->toGMM(initial_theta);

            // create training data set by concatenating X and Y matrices
            mat training_data = join_cols(x.t(),y.t());

            // train the GMM with the training data set
            gmm_full model;
            int n_gaus = M.n_cols;
            model.set_params(M, V, Rou.t());
            auto start = std::chrono::high_resolution_clock::now();

            if(model.learn(training_data, n_gaus, maha_dist, keep_existing, 0, config->em_iteration,1e-10 ,true)){
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
                cout << duration.count() << endl;
                //return this->fromGMM(x.n_cols, y.n_cols, n_gaus);
            }

            //else
            //    throw std::string("GMM learning failed");
        };

        mat getPosterior();
        void train(mat x, int nb_iteration);

    private:
        vec Rou;
        mat M;
        cube V;
        mat posterior;
        std::shared_ptr<GMMLearningConfig> config;

        GLLiMParameters<FullCovariance, FullCovariance> fromGMM(int K, int D, int L){

            GLLiMParameters<FullCovariance, FullCovariance> gLLiMParameters;

            gLLiMParameters.A = cube(D, L, K);
            gLLiMParameters.B = mat(D, K);
            gLLiMParameters.C = mat(L, K);
            gLLiMParameters.Pi = normalise(vec(K, fill::randu), 1);
            gLLiMParameters.Sigma = std::vector<FullCovariance>(K);
            gLLiMParameters.Gamma = std::vector<FullCovariance>(K);

            mat m_x = M.submat(0, 0, L-1, K-1);
            mat m_y = M.submat(L, 0, L+D-1, K-1);

            cube v_xx = V.subcube(0, 0, 0, L-1, L-1, K-1);
            cube v_xx_inv = v_xx;
            v_xx_inv.each_slice( [](mat& X){ X = inv(X); } );
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
        };

        void toGMM(std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> theta){
            int K = theta->Pi.n_rows;
            int L = theta->C.n_rows;
            int D = theta->B.n_rows;

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
                        join_rows(theta->Gamma[j] * mat(Rou.n_cols, Rou.n_cols, fill::eye),
                                theta->Gamma[j] * mat(theta->A.slice(j).t())),
                        join_rows(theta->A.slice(j) * theta->Gamma[j],
                                theta->Sigma[j] + theta->A.slice(j) * theta->Gamma[j] * theta->A.slice(j).t()));
            }
        };
    };

}

#endif //KERNELO_ESTIMATORS_H
