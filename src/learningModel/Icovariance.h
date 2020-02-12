//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_ICOVARIANCE_H
#define KERNELO_ICOVARIANCE_H

#include <armadillo>

using namespace arma;

namespace learningModel{

    //------------------ Icovariance ------------------------ //
    class Icovariance{};

    vec operator + (const Icovariance &x, const vec &y);


    //----------------- DiagCovariance ---------------------- //
    class DiagCovariance : public Icovariance{

    public:
        explicit DiagCovariance(const vec &covariance){
            this->covariance = covariance;
        }

        DiagCovariance()= default;;

        vec get_covariance() const{
            return covariance;
        }

        DiagCovariance &operator = (const DiagCovariance &cov){
            covariance = cov.get_covariance();
        }

    private:
        vec covariance;

    };


    vec operator + (const DiagCovariance &x, const vec &y){
        return y + x.get_covariance();
    }

    //----------------- FullCovariance ---------------------- //
    class FullCovariance : public Icovariance{

    public:
        explicit FullCovariance(const mat &covariance){
            this->covariance = covariance;
        }

        FullCovariance()= default;;

        mat get_covariance() const{
            return covariance;
        }

        FullCovariance &operator = (const DiagCovariance &cov){
            covariance = cov.get_covariance();
        }

    private:
        mat covariance;

    };

    //----------------- FullCovariance ---------------------- //
    class IsoCovariance : public Icovariance{

    public:
        explicit IsoCovariance(double covariance){
            this->covariance = covariance;
        }

        IsoCovariance()= default;;

        double get_covariance() const{
            return covariance;
        }

        IsoCovariance &operator = (const IsoCovariance &cov){
            covariance = cov.get_covariance();
        }

    private:
        double covariance;
    };


}

#endif //KERNELO_ICOVARIANCE_H
