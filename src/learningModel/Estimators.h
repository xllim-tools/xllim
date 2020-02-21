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
        virtual GLLiMParameters<T,U> estimate(const mat& x, const mat& y, GLLiMParameters<T,U> initial_theta) = 0;

    };

    template <typename T , typename U >
    class EmEstimator : public Iestimator<T,U>{

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        explicit EmEstimator(const std::shared_ptr<EMLearningConfig>& config){
            this->config = config;
        };

        void next_rnk(const mat& x, const mat& y, GLLiMParameters<T,U> theta, mat &next_rnk){
            int K = theta.Pi.n_rows;
            int L = theta.C.n_rows;
            int D = theta.B.n_rows;
            int N = x.n_rows;


            for(unsigned k=0; k<K; k++){
                //std::cout << "K : "<< k << std::endl;
                if(theta.Pi(k)){

                    double temp1 = D * log(2* datum::pi) + log(theta.Sigma[k].det());
                    double temp2 = L * log(2* datum::pi) + log(theta.Gamma[k].det());
                    FullCovariance sigma_inv = theta.Sigma[k].inv(false);
                    FullCovariance gamma_inv = theta.Gamma[k].inv(false);
                    vec test = y.row(0).t() - theta.A.slice(k)*x.row(0).t() - theta.B.col(k);
                    //sigma_inv.print();
                    //test.t().print();
                    theta.A.slice(k).print("A");
                    x.row(0).print("X");
                    theta.B.col(k).t().print("B");

                    for(unsigned n=0; n<N; n++ ){
                        vec y_u = y.row(n).t() - theta.A.slice(k)*x.row(n).t() - theta.B.col(k);
                        vec x_u = x.row(n).t() - theta.C.col(k);

                        next_rnk(n,k) = log(theta.Pi(k)) - 0.5 *
                                (temp1 +  dot(y_u ,sigma_inv * y_u)) - 0.5 *
                                (temp2 +  dot(x_u , gamma_inv * x_u));
                        /*if(next_rnk(n,k) == 0){
                            std::cout << "nan1 : "<< log(theta.Pi(k)) <<  std::endl;
                            std::cout << "nan2 : "<< log(theta.Sigma[k].det()) <<  std::endl;
                            std::cout << "nan3 : "<< log(theta.Gamma[k].det()) <<  std::endl;
                            std::cout << "nan4 : "<< dot(y_u ,sigma_inv * y_u) <<  std::endl;

                        }*/

                    }
                }


            }

            //next_rnk.row(0).print("next_rnk");

            double sum = 0;
            double alpha = 1;
            for(unsigned n=0; n<N; n++ ){
                sum = 0;
                if(exp(next_rnk.row(n).max()) != 0){
                    for(unsigned k=0; k<K; k++){
                        sum += exp(next_rnk(n,k) - next_rnk.row(n).max());
                    }
                    if(n==0)
                        std::cout << "sum : "<< sum <<  std::endl;
                    next_rnk.row(n) -= (log(sum) + next_rnk.row(n).max());
                }
            }

            //next_rnk.row(0).print("next_rnk");



        }

        void next_theta(const mat& x, const mat& y, const mat& r_nk, GLLiMParameters<T,U> &next_theta){
            int N = r_nk.n_rows;
            int K = r_nk.n_cols;
            int L = 6; // need fix
            int D = 50; // need fix
            double r_k = 0;

            for(unsigned k=0; k<K; k++){

                r_k = 0;

                for(unsigned n=0; n<N; n++ ){
                    r_k += exp(r_nk(n,k));// - r_nk.col(k).max());
                }
                //r_k -= (log(r_k) + r_nk.col(k).max());

                //std::cout << "r_k : "<< r_k << std::endl;


                // Update Pi
                //std::cout << "K : "<< k << "Pi" << std::endl;
                next_theta.Pi(k) = r_k/N;

                if(r_k != 0){
                    // Update C
                    //std::cout << "K : "<< k << "C" << std::endl;
                    next_theta.C.col(k) = 0.0;
                    for(unsigned n=0; n<N; n++) {
                        next_theta.C.col(k) += x.row(n).t() * exp(r_nk(n,k)) / r_k;
                    }


                    // Update Gamma
                    //std::cout << "K : "<< k << "Gamma" << std::endl;
                    next_theta.Gamma[k] = 0.0;
                    for(unsigned n=0; n<N; n++){
                        next_theta.Gamma[k] +=  (x.row(n).t() - next_theta.C.col(k)) * (x.row(n).t() - next_theta.C.col(k)).t() * exp(r_nk(n,k)) / r_k;
                    };

                    // Update A

                    mat X_k(L,N);
                    mat Y_k(D,N);
                    rowvec x_k = rowvec(next_theta.C.col(k).t()); //same formula
                    rowvec y_k(D, fill::zeros);
                    for(unsigned n=0; n<N; n++){
                        y_k += y.row(n) * exp(r_nk(n,k)) / r_k;
                    }

                    for(unsigned n=0; n<N; n++){
                        X_k.col(n) = sqrt(exp(r_nk(n,k)) / r_k) * (x.row(n)- x_k).t();
                        Y_k.col(n) = sqrt(exp(r_nk(n,k)) / r_k) * (y.row(n)- y_k).t();
                    }

                    //std::cout << "K : "<< k << "A" << std::endl;
                    next_theta.A.slice(k) = Y_k * X_k.t() * inv(X_k * X_k.t());

                    //update B
                    //std::cout << "K : "<< k << "B" << std::endl;
                    next_theta.B.col(k) = 0.0;
                    for(unsigned n=0; n<N; n++){
                        next_theta.B.col(k) += (y.row(n).t() - next_theta.A.slice(k) * x.row(n).t()) * exp(r_nk(n,k)) / r_k;
                    }

                    // Update Sigma
                    //std::cout << "K : "<< k << "Sigma" << std::endl;
                    next_theta.Sigma[k] = 0.0;
                    for(unsigned n=0; n<N; n++){
                        next_theta.Sigma[k] += (y.row(n).t() - next_theta.A.slice(k) * x.row(n).t() - next_theta.B.col(k))
                                               * (y.row(n).t() - next_theta.A.slice(k) * x.row(n).t() - next_theta.B.col(k)).t()
                                               * exp(r_nk(n,k)) / r_k;
                    }
                }


            }
            /*r_nk.col(0).t().print("r_nk");
            next_theta.C.col(0).t().print("C");
            next_theta.B.col(0).t().print("B");
            next_theta.A.slice(0).print("A");
            next_theta.Pi.t().print("Pi");
            next_theta.Gamma[0].print();
            next_theta.Sigma[0].print();*/

        }

        GLLiMParameters<T, U> estimate(
                const mat& x,
                const mat& y,
                GLLiMParameters<T, U> initial_theta) override{

            mat r_nk(x.n_rows, initial_theta.Pi.n_rows, fill::zeros);


            for(unsigned iter=0; iter<(config->max_iteration); iter++){
                std::cout << "iteration "<< iter << " : " << std::endl;
                next_rnk(x,y,initial_theta, r_nk);
                next_theta(x,y,r_nk,initial_theta);
                //r_nk.row(0).print("rnk_0");
                vec temp_dens(x.n_rows, fill::zeros);
                double result = 0;



                for(unsigned n=0; n<x.n_rows; n++){
                    double temp = 0;
                    for(unsigned k=0; k<initial_theta.Pi.n_rows; k++){
                        temp += exp(r_nk(n,k) - r_nk.row(n).max());
                    }
                    temp_dens(n) = log(temp) + r_nk.row(n).max();
                }


                for(unsigned n=0; n<x.n_rows; n++){
                    result += exp(temp_dens(n) - temp_dens.max());
                }
                result = log(result) + temp_dens.max() - log(x.n_rows);
                std::cout << "log_vraissamblance " << result << std::endl;
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

        GLLiMParameters<FullCovariance, FullCovariance> estimate(
                const mat& x,
                const mat& y,
                GLLiMParameters<FullCovariance, FullCovariance> initial_theta) override {
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
                return this->fromGMM(x.n_cols, y.n_cols, n_gaus);
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

        void toGMM(GLLiMParameters<FullCovariance, FullCovariance> theta){
            int K = theta.Pi.n_rows;
            int L = theta.C.n_rows;
            int D = theta.B.n_rows;

            // GMM weights
            this->Rou = theta.Pi;

            // GMM means
            mat AC = mat(D,K);
            for(unsigned i = 0; i < K; i++){
                AC.col(i) = theta.A.slice(i) * theta.C.col(i);
            }
            this->M = join_cols(theta.C, AC + theta.B);

            // GMM Covariances
            V = cube(D+L, D+L, K);
            for(unsigned j = 0; j < K; j++){
                V.slice(j) =join_cols(
                        join_rows(theta.Gamma[j] * mat(Rou.n_cols, Rou.n_cols, fill::eye),
                                theta.Gamma[j] * mat(theta.A.slice(j).t())),
                        join_rows(theta.A.slice(j) * theta.Gamma[j],
                                theta.Sigma[j] + theta.A.slice(j) * theta.Gamma[j] * theta.A.slice(j).t()));
            }
        };
    };

}

#endif //KERNELO_ESTIMATORS_H
