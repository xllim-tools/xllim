//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_ICOVARIANCE_H
#define KERNELO_ICOVARIANCE_H

#include <armadillo>

using namespace arma;

namespace learningModel{

    //------------------ Icovariance ------------------------ //
    class Icovariance{
    };


    //----------------- DiagCovariance ---------------------- //
    class DiagCovariance : public Icovariance{

        friend mat operator + (const mat &y, const DiagCovariance &x);
        friend mat operator + (const DiagCovariance &x, const mat &y);
        friend mat operator - (const mat &y, const DiagCovariance &x);
        friend mat operator - (const DiagCovariance &x, const mat &y);
        friend mat operator * (const mat &y, const DiagCovariance &x);
        friend mat operator * (const DiagCovariance &x, const mat &y);
        friend vec operator * (const DiagCovariance &x, const vec &y);
        friend rowvec operator * (const rowvec &y, const DiagCovariance &x);

    public:


        explicit DiagCovariance(const vec &covariance);
        explicit DiagCovariance(const mat &covariance);
        DiagCovariance(unsigned dimension);
        DiagCovariance() = default;
        DiagCovariance &operator = (const DiagCovariance &cov);
        DiagCovariance &operator = (const mat &cov);
        DiagCovariance &operator = (double scalar);
        DiagCovariance &operator += (const mat &cov);
        DiagCovariance &operator += (double scalar);
        DiagCovariance inv();
        void rankOneUpdate(const vec &v, double alpha);
        void print();
        double det();
        mat getFull() const;

    private:
        vec covariance;

    };



    //----------------- FullCovariance ---------------------- //
    class FullCovariance : public Icovariance{
        friend mat operator + (const mat &y, const FullCovariance &x);
        friend mat operator + (const FullCovariance &x, const mat &y);
        friend mat operator - (const mat &y, const FullCovariance &x);
        friend mat operator - (const FullCovariance &x, const mat &y);
        friend mat operator * (const mat &y, const FullCovariance &x);
        friend mat operator * (const FullCovariance &x, const mat &y);
        friend vec operator * (const FullCovariance &x, const vec &y);
        friend rowvec operator * (const rowvec &y, const FullCovariance &x);

    public:


        explicit FullCovariance(const mat &covariance);
        FullCovariance(unsigned dimension);
        FullCovariance() = default;
        FullCovariance &operator = (const FullCovariance &cov);
        FullCovariance &operator = (const mat &cov);
        FullCovariance &operator = (double scalar);
        FullCovariance &operator += (const mat &cov);
        FullCovariance &operator += (double scalar);
        FullCovariance inv();
        void rankOneUpdate(const vec &v, double alpha);
        void print();
        double det();
        mat getFull() const;

    private:
        mat covariance;
    };

    //----------------- IsoCovariance ---------------------- //
    class IsoCovariance : public Icovariance{

        friend mat operator + (const mat &y, const IsoCovariance &x);
        friend mat operator + (const IsoCovariance &x, const mat &y);
        friend mat operator - (const mat &y, const IsoCovariance &x);
        friend mat operator - (const IsoCovariance &x, const mat &y);
        friend mat operator * (const mat &y, const IsoCovariance &x);
        friend mat operator * (const IsoCovariance &x, const mat &y);
        friend vec operator * (const IsoCovariance &x, const vec &y);
        friend rowvec operator * (const rowvec &y, const IsoCovariance &x);

    public:

        explicit IsoCovariance(double covariance, unsigned size);
        explicit IsoCovariance(const mat &covariance);
        IsoCovariance(unsigned dimension);
        IsoCovariance() = default;
        IsoCovariance &operator = (const IsoCovariance &cov);
        IsoCovariance &operator = (const mat &cov);
        IsoCovariance &operator = (double scalar);
        IsoCovariance &operator += (const mat &cov);
        IsoCovariance &operator += (double scalar);
        IsoCovariance inv();
        void rankOneUpdate(const vec &v, double alpha);
        void print();
        double det();
        mat getFull() const;

    private:
        double covariance;
        unsigned size;
    };


}

#endif //KERNELO_ICOVARIANCE_H
