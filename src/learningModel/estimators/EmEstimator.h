//
// Created by reverse-proxy on 26‏/2‏/2020.
//

#ifndef KERNELO_EMESTIMATOR_H
#define KERNELO_EMESTIMATOR_H

#include "Estimators.h"

namespace learningModel{

    template <typename T , typename U >
    class EmEstimator : public Iestimator<T,U>{
        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        explicit EmEstimator(const std::shared_ptr<EMLearningConfig>& config);
        EmEstimator();
        void next_rnk(const mat& x, const mat& y, std::shared_ptr<GLLiMParameters<T, U>> theta, mat &next_rnk);
        void next_theta(const mat& x, const mat& y, const mat& r_nk, std::shared_ptr<GLLiMParameters<T, U>> next_theta);
        mat norm_log_rnk(const mat &r_nk);
        double log_likelihood(const mat& r_nk);

        void execute(
                const mat& x,
                const mat& y,
                std::shared_ptr<GLLiMParameters<T, U>> initial_theta) override;

    private:
        std::shared_ptr<EMLearningConfig> config;
        void update_A_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const mat &y, const vec &exp_avg_rnk);
        void update_B_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &Y_AX, const vec &exp_avg_rnk);
        void update_Sigma_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &Y_AX, const vec &exp_avg_rnk);
        void update_Gamma_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const vec &exp_avg_rnk);
        void update_C_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, const mat &x, const vec &exp_avg_rnk);
        void update_Pi_k(std::shared_ptr<GLLiMParameters<T, U>> &next_theta, unsigned k, unsigned N, double r_k);

        template <typename V>
        void covStabilityImprov(V &covariance, unsigned dimension, double floor);

    };

}

#include "EmEstimator.tpp"


#endif //KERNELO_EMESTIMATOR_H
