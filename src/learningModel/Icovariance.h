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

    private:
        vec covariance;

    };



    //----------------- FullCovariance ---------------------- //
    class FullCovariance : public Icovariance{

        friend mat operator + (const mat &y, const FullCovariance &x);
        friend mat operator + (const FullCovariance &x, const mat &y);
        friend mat operator * (const mat &y, const FullCovariance &x);
        friend mat operator * (const FullCovariance &x, const mat &y);

    public:
        FullCovariance(const mat &covariance);
        FullCovariance()= default;
        FullCovariance &operator = (const FullCovariance &cov);
        FullCovariance &operator = (const mat &cov);

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
