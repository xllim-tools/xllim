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


    //----------------- DiagCovariance ---------------------- //
    class DiagCovariance : public Icovariance{

        friend mat operator + (const mat &y, const DiagCovariance &x);
        friend mat operator + (const DiagCovariance &x, const mat &y);
        friend mat operator * (const mat &y, const DiagCovariance &x);
        friend mat operator * (const DiagCovariance &x, const mat &y);

    public:
        explicit DiagCovariance(const vec &covariance);
        DiagCovariance()= default;
        DiagCovariance &operator = (const DiagCovariance &cov);
        DiagCovariance &operator = (const mat &cov);
        DiagCovariance inv();
        double det();

    private:
        vec covariance;

    };



    //----------------- FullCovariance ---------------------- //
    class FullCovariance : public Icovariance{

        friend mat operator + (const mat &y, const FullCovariance &x);
        friend mat operator + (const FullCovariance &x, const mat &y);
        friend mat operator * (const mat &y, const FullCovariance &x);
        friend mat operator * (const FullCovariance &x, const mat &y);
        friend vec operator * (const FullCovariance &x, const vec &y);
        friend rowvec operator * (const rowvec &y, const FullCovariance &x);


    public:
        explicit FullCovariance(const mat &covariance);
        FullCovariance()= default;
        FullCovariance &operator = (const FullCovariance &cov);
        FullCovariance &operator = (const mat &cov);
        FullCovariance &operator = (double scalar);
        FullCovariance &operator += (const mat &cov);
        FullCovariance &operator += (double scalar);
        FullCovariance inv(bool print);
        void rankOneUpdate(const vec &v, double alpha);
        void print();
        double det();

    private:
        mat covariance;
    };

    //----------------- IsoCovariance ---------------------- //
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
