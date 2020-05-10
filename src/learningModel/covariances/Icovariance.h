/**
 * @file Icovariance.h
 * @brief The classes of the different type of covariance matrices
 * @author Sami DJOUADI
 * @version 1.1
 * @date 12/02/2020
 */

#ifndef KERNELO_ICOVARIANCE_H
#define KERNELO_ICOVARIANCE_H

#include <armadillo>
#include "../../helpersFunctions/Helpers.h"

using namespace arma;

namespace learningModel{

    /**
     * @class Icovariance
     * @brief This interface has been created for generalisation purpose
     */
    class Icovariance{};

    /**
     * @class DiagCovariance
     * @brief This class stands for a covariance matrix where all the covariances are put to zero.
     *
     * @details In order to faster the execution time and reduce the memory size needed to store this type of covariance
     * matrix, this class stores only the diagonal of the matrix as a vector. It overrides all the basic operations on
     * matrices like addition and multiplication, the computation of the determinant and the inverse of the matrix.
     *
     */
    class DiagCovariance : public Icovariance{

        /**
         * @brief Addition operator redefinition
         * @details The method performs C = A + B where C and A are armadillo matrices and B is of type DiagCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : DiagCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator + (const mat &y, const DiagCovariance &x);

        /**
         * @brief Addition operator redefinition
         * @details The method performs C = A + B where C and B are armadillo matrices and A is of type DiagCovariance.
         * @param x : DiagCovariance
         * @param y : armadillo::Mat<double>
         * @return : armadillo::Mat<double>
         */
        friend mat operator + (const DiagCovariance &x, const mat &y);

        /**
         * @brief Subtraction operator redefinition
         * @details The method performs C = A - B where C and A are armadillo matrices and B is of type DiagCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : DiagCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator - (const mat &y, const DiagCovariance &x);

        /**
         * @brief Subtraction operator redefinition
         * @details The method performs C = A - B where C and B are armadillo matrices and A is of type DiagCovariance.
         * @param x : DiagCovariance
         * @param y : armadillo::Mat<double>
         * @return : armadillo::Mat<double>
         */
        friend mat operator - (const DiagCovariance &x, const mat &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and A are armadillo matrices and B is of type DiagCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : DiagCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator * (const mat &y, const DiagCovariance &x);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and B are armadillo matrices and A is of type DiagCovariance.
         * @param x : DiagCovariance
         * @param y : armadillo::Mat<double>
         * @return : armadillo::Mat<double>
         */
        friend mat operator * (const DiagCovariance &x, const mat &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and B are column vectors and A is of type DiagCovariance.
         * @param y : armadillo::Col<double>
         * @param x : DiagCovariance
         * @return : armadillo::Col<double>
         */
        friend vec operator * (const DiagCovariance &x, const vec &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and A are row vectors and A is of type DiagCovariance.
         * @param y : armadillo::Row<double>
         * @param x : DiagCovariance
         * @return : armadillo::Row<double>
         */
        friend rowvec operator * (const rowvec &y, const DiagCovariance &x);

    public:
        /**
         * @brief Constructor
         * @param covariance : a vector of variances
         */
        explicit DiagCovariance(const vec &covariance);

        /**
         * @biref Constructor
         * @param covariance : a matrix of covariance that only its diagonal is considered.
         */
        explicit DiagCovariance(const mat &covariance);

        /**
         * @brief Constructor
         * @param dimension : the dimension of the empty vector of variances.
         */
        DiagCovariance(unsigned dimension);

        /**
         * @brief Constructor
         */
        DiagCovariance() = default;

        /**
         * @brief Assignment operator redefinition
         * @param cov : a reference to a DiagCovariance object
         * @return DiagCovariance
         */
        DiagCovariance &operator = (const DiagCovariance &cov);

        /**
         * @brief Assignment operator redefinition
         * @details This method copies the diagonal of the matrix of covariances to the current DiagCovariance object
         * @param cov : a reference to an armadillo matrix
         * @return DiagCovariance
         */
        DiagCovariance &operator = (const mat &cov);

        /**
         * @brief Assignment operator redefinition
         * @details This method replaces the values of the vector of variances of the current DiagCovariance object
         * with the parameter of the constructor.
         * @param scalar : double
         * @return DiagCovariance
         */
        DiagCovariance &operator = (double scalar);

        /**
         * @brief Assignment addition operator redefinition
         * @details This method adds the diagonal of the matrix in parameter to the vector of variances of the current
         * DiagCovariance object.
         * @param cov : a reference to an armadillo matrix
         * @return DiagCovariance
         */
        DiagCovariance &operator += (const mat &cov);

        /**
         * @brief Assignment addition operator redefinition
         * @details This method adds the parameter of the constructor to all the variances of the current DiagCovariance object.
         * @param scalar
         * @return DiagCovariance
         */
        DiagCovariance &operator += (double scalar);

        /**
         * @brief The method Computes the inverse of the covariance matrix
         * @return DiagCovariance
         */
        DiagCovariance inv();

        /**
         * @brief rankOneUpdate
         * @details If A is a vector of variances, B a column vector and alpha a scalar, this methods computes :
         * A = A + B^2 * alpha
         * @param v
         * @param alpha
         */
        void rankOneUpdate(const vec &v, double alpha);

        /**
         * @brief The method prints the vector of variances.
         */
        void print();

        /**
         * @brief The method computes the determinant of the matrix of covariances with zero covariances.
         * @details The determinant is the product of the variances.
         * @return double : determinant
         */
        double det();

        /**
         * @brief Creates an armadillo matrix with zero covariances and variances from the current DiagCovairance object
         * @return armadillo::Mat<double>
         */
        mat getFull() const;

    private:
        vec variances; /**< The vector of variances */
    };

    /**
     * @class FullCovariance
     * @brief This class stands for a covariance matrix with no constraint.
     *
     */
    class FullCovariance : public Icovariance{
        /**
         * @brief Addition operator redefinition
         * @details The method performs C = A + B where C and A are armadillo matrices and B is of type FullCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : FullCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator + (const mat &y, const FullCovariance &x);

        /**
         * @brief Addition operator redefinition
         * @details The method performs C = A + B where C and B are armadillo matrices and A is of type FullCovariance.
         * @param x : FullCovariance
         * @param y : armadillo::Mat<double>
         * @return : armadillo::Mat<double>
         */
        friend mat operator + (const FullCovariance &x, const mat &y);

        /**
         * @brief Subtraction operator redefinition
         * @details The method performs C = A - B where C and A are armadillo matrices and B is of type FullCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : FullCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator - (const mat &y, const FullCovariance &x);

        /**
         * @brief Subtraction operator redefinition
         * @details The method performs C = A - B where C and B are armadillo matrices and A is of type FullCovariance.
         * @param x : FullCovariance
         * @param y : armadillo::Mat<double>
         * @return : armadillo::Mat<double>
         */
        friend mat operator - (const FullCovariance &x, const mat &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and A are armadillo matrices and B is of type FullCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : FullCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator * (const mat &y, const FullCovariance &x);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and B are armadillo matrices and A is of type FullCovariance.
         * @param x : FullCovariance
         * @param y : armadillo::Mat<double>
         * @return : armadillo::Mat<double>
         */
        friend mat operator * (const FullCovariance &x, const mat &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and B are column vectors and A is of type FullCovariance.
         * @param y : armadillo::Col<double>
         * @param x : FullCovariance
         * @return : armadillo::Col<double>
         */
        friend vec operator * (const FullCovariance &x, const vec &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and A are row vectors and A is of type FullCovariance.
         * @param y : armadillo::Row<double>
         * @param x : FullCovariance
         * @return : armadillo::Row<double>
         */
        friend rowvec operator * (const rowvec &y, const FullCovariance &x);

    public:
        /**
         * @biref Constructor
         * @param covariance : armadillo:Mat<double>.
         */
        explicit FullCovariance(const mat &covariance);

        /**
         * @brief Constructor
         * @param dimension : the dimension of the matrix of covariance.
         */
        FullCovariance(unsigned dimension);

        /**
         * @brief Constructor
         */
        FullCovariance() = default;

        /**
         * @brief Assignment operator redefinition
         * @param cov : a reference to a FullCovariance object
         * @return FullCovariance
         */
        FullCovariance &operator = (const FullCovariance &cov);

        /**
         * @brief Assignment operator redefinition
         * @details This method replaces the matrix of covariances of the current FullCovariance object with the one in parameter.
         * @param cov : a reference to an armadillo matrix
         * @return FullCovariance
         */
        FullCovariance &operator = (const mat &cov);

        /**
         * @brief Assignment operator redefinition
         * @details This method replaces the values of the matrix of covariances of the current FullCovariance object
         * with the parameter of the constructor.
         * @param scalar : double
         * @return FullCovariance
         */
        FullCovariance &operator = (double scalar);

        /**
         * @brief Assignment addition operator redefinition
         * @details This method adds the matrix in parameter to the matrix of covariances of the current FullCovariance object.
         * @param cov : a reference to an armadillo matrix
         * @return DiagCovariance
         */
        FullCovariance &operator += (const mat &cov);

        /**
         * @brief Assignment addition operator redefinition
         * @details This method adds the parameter of the constructor to all the values of the matrix of the current FullCovariance object.
         * @param scalar
         * @return FullCovariance
         */
        FullCovariance &operator += (double scalar);

        /**
         * @brief The method Computes the inverse of the covariance matrix.
         * @return FullCovariance
         */
        FullCovariance inv();

        /**
         * @brief rankOneUpdate
         * @details If A is a matrix of covariances, B a column vector and alpha a scalar, this methods computes :
         * A = A + B * B.transpose() * alpha
         * @param v
         * @param alpha
         */
        void rankOneUpdate(const vec &v, double alpha);

        /**
         * @brief The method prints the matrix of covariances.
         */
        void print();

        /**
         * @brief The method computes the determinant of the matrix of covariances.
         * @return double : determinant
         */
        double det();

        /**
         * @brief Returns the matrix of covariance in armadillo data structure
         * @return armadillo::Row<double>
         */
        mat getFull() const;

    private:
        mat covariance; /**< The matrix of covariance. */
    };

    /**
    * @class IsoCovariance
    * @brief This class stands for a covariance matrix where all the covariances are put to zero and all the variances
    * are equal.
    *
    * @details In order to faster the execution time and reduce the memory size needed to store this type of covariance
    * matrix, this class stores only the variance from the matrix as a scalar. It overrides all the basic operations on
    * matrices like addition and multiplication, the computation of the determinant and the inverse of the matrix.
    *
    */
    class IsoCovariance : public Icovariance{
        /**
         * @brief Addition operator redefinition
         * @details The method performs C = A + B where C and A are armadillo matrices and B is of type IsoCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : IsoCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator + (const mat &y, const IsoCovariance &x);

        /**
        * @brief Addition operator redefinition
        * @details The method performs C = A + B where C and B are armadillo matrices and A is of type IsoCovariance.
        * @param x : IsoCovariance
        * @param y : armadillo::Mat<double>
        * @return : armadillo::Mat<double>
        */
        friend mat operator + (const IsoCovariance &x, const mat &y);

        /**
         * @brief Subtraction operator redefinition
         * @details The method performs C = A - B where C and A are armadillo matrices and B is of type IsoCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : IsoCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator - (const mat &y, const IsoCovariance &x);

        /**
         * @brief Subtraction operator redefinition
         * @details The method performs C = A - B where C and B are armadillo matrices and A is of type IsoCovariance.
         * @param x : IsoCovariance
         * @param y : armadillo::Mat<double>
         * @return : armadillo::Mat<double>
         */
        friend mat operator - (const IsoCovariance &x, const mat &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and A are armadillo matrices and B is of type IsoCovariance.
         * @param y : armadillo::Mat<double>
         * @param x : IsoCovariance
         * @return : armadillo::Mat<double>
         */
        friend mat operator * (const mat &y, const IsoCovariance &x);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and B are armadillo matrices and A is of type IsoCovariance.
         * @param x : IsoCovariance
         * @param y : armadillo::Mat<double>
         * @return : armadillo::Mat<double>
         */
        friend mat operator * (const IsoCovariance &x, const mat &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and B are column vectors and A is of type IsoCovariance.
         * @param y : armadillo::Col<double>
         * @param x : IsoCovariance
         * @return : armadillo::Col<double>
         */
        friend vec operator * (const IsoCovariance &x, const vec &y);

        /**
         * @brief Multiplication operator redefinition
         * @details The method performs C = A * B where C and A are row vectors and A is of type IsoCovariance.
         * @param y : armadillo::Row<double>
         * @param x : IsoCovariance
         * @return : armadillo::Row<double>
         */
        friend rowvec operator * (const rowvec &y, const IsoCovariance &x);

    public:

        /**
         * Constructor
         * @param scalar : is the value of the variance of all the variables
         * @param size : the number of variables
         */
        explicit IsoCovariance(double scalar, unsigned size);

        /**
         * @biref Constructor
         * @param covariance : armadillo:Mat<double>.
         */
        explicit IsoCovariance(const mat &covariance);

        /**
         * @brief Constructor
         * @param dimension : the dimension of the matrix of covariance.
         */
        IsoCovariance(unsigned dimension);

        /**
         * Constructor
         */
        IsoCovariance() = default;

        /**
         * @brief Assignment operator redefinition
         * @param cov : a reference to a IsoCovariance object
         * @return IsoCovariance
         */
        IsoCovariance &operator = (const IsoCovariance &cov);

        /**
         * @brief Assignment operator redefinition
         * @details This method replaces the of the scalar of the current object by the mean of the diagonal of the matrix in parameter.
         * @param cov : a reference to an armadillo matrix
         * @return IsoCovariance
         */
        IsoCovariance &operator = (const mat &cov);

        /**
         * @brief Assignment operator redefinition
         * @param scalar : double
         * @return IsoCovariance
         */
        IsoCovariance &operator = (double scalar);

        /**
         * @brief Assignment addition operator redefinition
         * @details This method adds the mean of the diagonal of the matrix in parameter to the scalar of the current
         * IsoCovariance object.
         * @param cov : a reference to an armadillo matrix
         * @return IsoCovariance
         */
        IsoCovariance &operator += (const mat &cov);

        /**
         * @brief Assignment addition operator redefinition
         * @details This method adds the parameter of the constructor to scalar of the current IsoCovariance object.
         * @param scalar
         * @return IsoCovariance
         */
        IsoCovariance &operator += (double scalar);

        /**
         * @brief The method Computes the inverse of the covariance matrix
         * @return IsoCovariance
         */
        IsoCovariance inv();

        /**
         * @brief rankOneUpdate
         * @details If A is a scalar, B a column vector and alpha a scalar, this methods computes :
         * A = A + mean(B^2 * alpha)
         * @param v
         * @param alpha
         */
        void rankOneUpdate(const vec &v, double alpha);

        /**
         * @brief The method prints the variance and the dimension of the equivalent full matrix of covariances.
         */
        void print();

        /**
         * @brief The method computes the determinant of the matrix of covariances with zero covariances and equal variances.
         * @details The determinant is equal to variance^dimension.
         * @return double : determinant
         */
        double det();

        /**
         * @brief Creates an armadillo matrix with zero covariances and equal variances from the current IsoCovariance object
         * @return armadillo::Mat<double>
         */
        mat getFull() const;

    private:
        double scalar; /**< the value of the variance of the variables*/
        unsigned size; /**< The number of variables*/
    };

    enum CovarianceType{
        FULL,
        DIAG,
        ISO
    };


}

#endif //KERNELO_ICOVARIANCE_H
