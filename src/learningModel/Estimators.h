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

        }

        void next_theta(const mat& x, const mat& y, const mat& rnk, GLLiMParameters<T,U> &next_theta){

        }

        GLLiMParameters<T, U> estimate(
                const mat& x,
                const mat& y,
                GLLiMParameters<T, U> initial_theta) override{

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
                                theta.Gamma[j] * theta.A.slice(j).t()),
                        join_rows(theta.A.slice(j) * theta.Gamma[j],
                                theta.Sigma[j] + theta.A.slice(j) * theta.Gamma[j] * theta.A.slice(j).t()));
            }
        };
    };

}

#endif //KERNELO_ESTIMATORS_H
