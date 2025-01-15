// /usr/bin/g++ -fdiagnostics-color=always -g test_perf_log_p.cpp ../../src/utils/utils.cpp -o test_perf_log_p.out -I/usr/include/python3.10 -I/home/luc/.local/lib/python3.10/site-packages/numpy/core/include/ -lpython3.10 -larmadillo -llapack -lblas -O0

#include "../../src/utils/utils.hpp"
#include <armadillo>
#include <omp.h>

using namespace arma;

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    double duration_arma(0);
    double duration_hand(0);
    std::vector<int> count_list = {1};//0, 1000};//, 50000, 1000000};
    int count;
    unsigned L=100, K=50, N=100000;
    std::cout << "L=" + std::to_string(L) + "; K=" + std::to_string(K) + "; N=" + std::to_string(N) << std::endl;

    for (size_t n = 0; n < count_list.size(); n++)
    {
        count = count_list[n];

        mat mean(L, K, fill::randu);

        // TODO full/diag
        // cube covariance(L, L, K, fill::value(0.01));
        // for (size_t i = 0; i < covariance.n_slices; ++i)
        // {
        //     // covariance.slice(i) *= covariance.slice(i).t();
        //     covariance.slice(i) += mat(L,L, fill::eye) * 0.1;
        //     // covariance(0,1,i) = 20;
        //     covariance.slice(i) *= covariance.slice(i).t();
        // }

        mat covariance(L, K, fill::randu);
        for (size_t i = 0; i < covariance.n_cols; ++i)
        {
            // covariance.slice(i) *= covariance.slice(i).t();
            covariance.col(i) += vec(L, fill::ones) * 0.1;
            // covariance(0,1,i) = 20;
            covariance.col(i) %= covariance.col(i);
        }


        // covariance.print("cov");
        // covariance.each_slice() *= covariance.each_slice().t();
        // rowvec weight(K, fill::value(0.19));
        // weight(1) = 0.24;
        rowvec weight(K, fill::randu);
        weight /= accu(weight);
        // weight = {0.4, 0.6};
        mat x(L, N, fill::randu);
        rowvec arma_log_p;
        rowvec hand_log_p;

        // TODO full/diag
        // gmm_full gmm;
        gmm_diag gmm;

        gmm.set_params(mean, covariance, weight);

        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < count; i++)
        {
            arma_log_p = gmm.log_p(x);
        }
        end = std::chrono::high_resolution_clock::now();
        duration_arma += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "arma time = " + std::to_string(duration_arma) << std::endl;
        // arma_log_p.print("arma_log_p");

        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < count; i++)
        {
            hand_log_p = utils::logSumExp(utils::logDensity(x, weight, mean, covariance), 1).t();
        }
        end = std::chrono::high_resolution_clock::now();
        duration_hand += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "hand time = " + std::to_string(duration_hand) << std::endl;
        // hand_log_p.print("hand_log_p");

        std::cout << "\nRATIO hand/arma" << std::endl;
        std::cout << std::to_string(duration_hand / duration_arma) << std::endl;
        std::cout << "\n" << std::endl;
    }
    return 0;
}