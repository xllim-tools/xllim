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

template <typename T , typename U >
void EmEstimator<T,U>::estimate(const mat &x, const mat &y, std::shared_ptr<GLLiMParameters<T, U>> initial_theta) {
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
    double max_rnk_row = 0;

    T sigma_inv;
    U gamma_inv;

    double det_sigma;
    double det_gamma;

    mat y_u(D, N, fill::zeros);
    vec x_u(L, fill::zeros);


    // OPTIM : compute matrix version of y_u and x_u


//#pragma omp parallel for shared(N,K,L,D,x,y,theta,D_log_2_pi, L_log_2_pi,temp_density_y,temp_density_x,log_Pi_K,next_rnk)
    for(unsigned k=0; k<K; k++){

        det_sigma = theta->Sigma[k].det();
        det_gamma = theta->Gamma[k].det();
        if(det_sigma != 0 && det_gamma != 0){
            if(theta->Pi(k) != 0){
                y_u = y - theta->A.slice(k) * x;
                y_u.each_col() -= theta->B.col(k);

                x_u = x;
                x_u.each_col() -= theta->C.col(k);

                temp_density_y = D_log_2_pi + log(det_sigma);
                temp_density_x = L_log_2_pi + log(det_gamma);
                sigma_inv = theta->Sigma[k].inv();
                gamma_inv = theta->Gamma[k].inv();
                log_Pi_K = log(theta->Pi(k));

                //sigma_inv.print();
                //gamma_inv.print();

                //std::cout << (temp_density_y ) << std::endl;

                for(unsigned n=0; n<N; n++ ){

                    next_rnk(n,k) = log_Pi_K -
                                    0.5 * (temp_density_y +  dot((rowvec(y_u.col(n).t()) * sigma_inv).t() , y_u.col(n))) -
                                    0.5 * (temp_density_x +  dot((rowvec(x_u.col(n).t()) * gamma_inv).t() , x_u.col(n)));

                    if(next_rnk(n,k) == (datum::inf)){
                        next_rnk(n,k) = -datum::inf;
                    }
                    if(n==0){
                        theta->Sigma[k].print();
                        sigma_inv.print();
                        std::cout << "nan1 : " << log_Pi_K << std::endl;
                        std::cout << "nan2 : " << theta->Sigma[k].det() << std::endl;
                        std::cout << "nan3 : " << theta->Gamma[k].det() << std::endl;
                        std::cout << "nan4 : " << y_u(11,n) << std::endl;
                        rowvec(y_u.col(n).t()).print("nan5");


                    }
                }

                //std::cout << "max rnk 0 : " <<next_rnk.row(0).max() << std::endl;
            }
        }else{
            next_rnk.col(k).fill(-datum::inf);
        }


    }


    next_rnk.t().print("rnk");

    // OPTIM : Open MP map reduce

    double log_l = 0;

//#pragma omp parallel for shared(N,K,next_rnk) private(max_rnk_row) schedule(dynamic) reduction(+:log_l)
    for(unsigned n=0; n<N; n++ ){
        double result = 0;
        max_rnk_row = next_rnk.row(n).max();
        for(unsigned k=0; k<K; k++){
            result += exp(next_rnk(n,k) - max_rnk_row);
        }
        log_l += (log(result) + max_rnk_row);
    }

    std::cout << "log_vraissamblance " << log_l/N << std::endl;

    // OPTIM : Open MP map reduce
    double sum = 0;

    for(unsigned n=0; n<N; n++ ){
        sum = 0;
        max_rnk_row = next_rnk.row(n).max();
        if(max_rnk_row != (-datum::inf)){
            for(unsigned k=0; k<K; k++){
                sum += exp(next_rnk(n,k) - max_rnk_row);
                if(n==0){
                    std::cout << "sum " << sum << std::endl;
                }
            }
            //std::cout << "deno " << (log(sum) + max_rnk_row) << std::endl;
            next_rnk.row(n) -= (log(sum) + max_rnk_row);
        }
    }

    next_rnk.t().print("rnk");


    //next_rnk.row(0).print("norm_log_rnk");

}

template <typename T , typename U >
void EmEstimator<T,U>::next_theta(const mat &x, const mat &y, const mat &r_nk,
                             std::shared_ptr <GLLiMParameters<T, U>> &next_theta) {

    int N = r_nk.n_rows;
    int K = r_nk.n_cols;
    int L = x.n_rows;
    int D = y.n_rows;


    mat X_k(L,N);
    mat Y_k(D,N);
    mat Y_AX(D,N);
    vec temp_sigma(D);
    vec temp_Gamma(L);
    vec exp_avg_rnk(N);
    vec y_k(D);
    double max_rnk_col = 0;
    double r_k = 0;

    //auto start1 = std::chrono::high_resolution_clock::now();



    /*auto end1 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    //cout << " update C: " <<duration0.count() << endl;

    start1 = std::chrono::high_resolution_clock::now();
    end1 = std::chrono::high_resolution_clock::now()*/;


    /*auto duration7 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto duration6 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);*/




//#pragma omp parallel for shared(x, y, r_nk, next_theta, N, K, D, L) schedule(dynamic)
    for(unsigned k=0; k<K; k++){

        r_k = 0;

        //start1 = std::chrono::high_resolution_clock::now();



        max_rnk_col = r_nk.col(k).max();

        if(max_rnk_col != (-datum::inf)){
            for(unsigned n=0; n<N; n++ ){
                r_k += exp(r_nk(n,k) - max_rnk_col);
            }
            r_k = (log(r_k) + max_rnk_col);
            exp_avg_rnk = exp(r_nk.col(k) - r_k);
        }else{
            r_k = -datum::inf;
        }



        //std:cout << "r_k : " << r_k << std::endl;

        //exp_avg_rnk.print("exp_avg_rnk");

        // Update Pi
        next_theta->Pi(k) = exp(r_k)/N;


        /*end1 = std::chrono::high_resolution_clock::now();
        duration7 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);*/

        if(exp(r_k) != 0){
            // Update C
            //start1 = std::chrono::high_resolution_clock::now();

            next_theta->C.col(k).fill(0.0);
            for(unsigned n=0; n<N; n++) {
                next_theta->C.col(k) += x.col(n) * exp_avg_rnk(n);
            }


            //next_theta->C.col(k).t().print("C");

            /*end1 = std::chrono::high_resolution_clock::now();
            duration1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);*/


            // Update Gamma
            //start1 = std::chrono::high_resolution_clock::now();


            next_theta->Gamma[k] = 0.0;
            for(unsigned n=0; n<N; n++){
                temp_Gamma = x.col(n) - next_theta->C.col(k);
                next_theta->Gamma[k].rankOneUpdate(temp_Gamma, exp_avg_rnk(n));
            }
            next_theta->Gamma[k] += eye(L,L) * 1e-8;

            //next_theta->Gamma[k].print();





            /*end1 = std::chrono::high_resolution_clock::now();
            duration2 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);*/
            //cout << "K : " << k << " update Gamma: " <<duration1.count() << endl;


            // Update A
            //start1 = std::chrono::high_resolution_clock::now();


            y_k.fill(0);
            for(unsigned n=0; n<N; n++){
                y_k += y.col(n) * exp_avg_rnk(n);
            }
            //y_k.print("y_k");
            //y.print("y");

            for(unsigned n=0; n<N; n++){
                X_k.col(n) = sqrt(exp_avg_rnk(n)) * (x.col(n)- next_theta->C.col(k));
                Y_k.col(n) = sqrt(exp_avg_rnk(n)) * (y.col(n)- y_k);
            }

            //Y_k.row(0).print("Y_k");
            (exp_avg_rnk.t()).print("Y_k");
            /*pinv(X_k * X_k.t()).print("pinv");
            std::cout << "nan : " << Y_k(0,0) << std::endl;*/

            if( accu(Y_k) != 0 && accu(X_k) != 0){
                next_theta->A.slice(k) = Y_k * X_k.t() * pinv(X_k * X_k.t());
            }

            next_theta->A.slice(k).print("A");




            /*end1 = std::chrono::high_resolution_clock::now();
            duration3 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);*/
            //cout << "K : " << k << " update A: " <<duration1.count() << endl;

            //start1 = std::chrono::high_resolution_clock::now();

            Y_AX = y - next_theta->A.slice(k) * x;

            //Y_AX.print("Y_AX");


            /*end1 = std::chrono::high_resolution_clock::now();
            duration4 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);*/
            //cout << "K : " << k << " Y_AX: " <<duration1.count() << endl;

            //update B
            //start1 = std::chrono::high_resolution_clock::now();

            next_theta->B.col(k).fill(0.0);
            for(unsigned n=0; n<N; n++){
                next_theta->B.col(k) += Y_AX.col(n) * exp_avg_rnk(n);
            }

            //next_theta->B.col(k).t().print("B");

            /*end1 = std::chrono::high_resolution_clock::now();
            duration5 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);*/
            //cout << "K : " << k << " Update B: " <<duration1.count() << endl;

            // Update Sigma
            //start1 = std::chrono::high_resolution_clock::now();


            next_theta->Sigma[k] = 0.0;
            //std::cout << (Y_AX(0,0) - next_theta->B(0,k)) << std::endl;
            for(unsigned n=0; n<N; n++){
                temp_sigma = Y_AX.col(n) - next_theta->B.col(k);
                next_theta->Sigma[k].rankOneUpdate(temp_sigma, exp_avg_rnk(n));
            }

            // Add noise to Sigma for computing stability
            next_theta->Sigma[k] += eye(D,D) * 1e-8;


            next_theta->Sigma[k].print();

            /*end1 = std::chrono::high_resolution_clock::now();
            duration6 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);*/
            //cout << "K : " << k << " update Sigma : " <<duration1.count() << endl;

            //std::cout << omp_get_thread_num() << "k : " << k << std::endl;

        }
    }

    /*cout << " update C: " <<duration1.count() << endl;
    cout << " update Gamma: " <<duration2.count() << endl;
    cout << " update A: " <<duration3.count() << endl;
    cout << " Y_AX: " <<duration4.count() << endl;
    cout << " update B: " <<duration5.count() << endl;
    cout << " update Sigma: " <<duration6.count() << endl;
    cout << " Rk and Pi: " <<duration7.count() << endl;*/

}