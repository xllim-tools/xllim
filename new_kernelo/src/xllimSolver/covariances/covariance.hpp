#ifndef COVARIANCE_HPP
#define COVARIANCE_HPP

#include <armadillo>

using namespace arma;

/**
 * @class Icovariance
 * @brief This interface has been created for generalisation purpose
 */
class Covariance // TODO useful interface ?
{
public:
    virtual double log_det() const = 0;
    // virtual Covariance inv() const = 0; // error: invalid covariant return type
    virtual void rank_one_update(const vec &v, double alpha) = 0;
    virtual void fill(const double scalar) = 0;
    virtual void print(const std::string &str) const = 0;
    virtual void print() const = 0;
    virtual mat get_mat() const = 0;
};

/**
 * @class FullCovariance
 * @brief This class stands for a covariance matrix with no constraint.

 * @details The class stores the symmetric positive definite covariance matrix as a mat object. In order to faster the
 * execution time it overrides all the basic operations on
 * matrices like addition and multiplication, the computation of the determinant and the inverse of the matrix.
 *
 */
class FullCovariance : public Covariance
{
public:
    // Constructors
    explicit FullCovariance(const mat &cov);
    explicit FullCovariance(unsigned dimension);
    FullCovariance() = default;

    // Class methods
    double log_det() const override;
    FullCovariance inv() const;
    void rank_one_update(const vec &v, double alpha) override;
    void fill(const double scalar) override;
    void print(const std::string &str) const override;
    void print() const override;
    FullCovariance head(unsigned L_t) const;
    FullCovariance tail(unsigned L_w) const;
    mat get_mat() const override;
    mat get() const { return get_mat(); };

    // Useful for conversion std::vector<Covariance> <=> Armadillo
    using Type = cube;
    static cube getTypeSize(unsigned K, unsigned dimension) { return cube(K, dimension, dimension); };

    // Assignement operators
    FullCovariance &operator=(const FullCovariance &cov);
    FullCovariance &operator=(const mat &cov);
    FullCovariance &operator+=(const mat &cov);
    FullCovariance &operator+=(double scalar);

private:
    mat covariances;
};

/**
 * @class DiagCovariance
 * @brief This class stands for a covariance matrix where all the covariances are put to zero except for the variances on the diagonal.
 *
 * @details In order to faster the execution time and reduce the memory size needed to store this type of covariance
 * matrix, this class stores only the diagonal of the matrix as a vector. It overrides all the basic operations on
 * matrices like addition and multiplication, the computation of the determinant and the inverse of the matrix.
 *
 */
class DiagCovariance : public Covariance
{
public:
    // Constructors
    explicit DiagCovariance(const vec &var);
    explicit DiagCovariance(const arma::subview_col<double> &var);
    explicit DiagCovariance(const arma::subview_row<double> &var);
    explicit DiagCovariance(const mat &cov);
    explicit DiagCovariance(unsigned dimension);
    DiagCovariance() = default;

    // Class methods
    double log_det() const override;
    DiagCovariance inv() const;
    void rank_one_update(const vec &v, double alpha) override;
    void fill(const double scalar) override;
    void print(const std::string &str) const override;
    void print() const override;
    DiagCovariance head(unsigned L_t) const;
    DiagCovariance tail(unsigned L_w) const;
    mat get_mat() const override;
    vec get_vec() const;
    rowvec get() const { return get_vec().t(); };

    // Useful for conversion std::vector<Covariance> <=> Armadillo
    using Type = mat;
    static mat getTypeSize(unsigned K, unsigned dimension) { return mat(K, dimension); };

    // Assignement operators
    DiagCovariance &operator=(const DiagCovariance &cov);
    DiagCovariance &operator=(const mat &cov);
    DiagCovariance &operator=(const vec &var);
    DiagCovariance &operator+=(const mat &cov);
    DiagCovariance &operator+=(double scalar);

private:
    vec variances;
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
class IsoCovariance : public Covariance
{
public:
    // Constructors
    explicit IsoCovariance(double variance, unsigned dimension);
    explicit IsoCovariance(const mat &covariance);
    explicit IsoCovariance(unsigned dimension);
    IsoCovariance() = default;

    // Class methods
    double log_det() const override;
    IsoCovariance inv() const;
    void rank_one_update(const vec &v, double alpha) override;
    void fill(const double scalar) override;
    void print(const std::string &str) const override;
    void print() const override;
    IsoCovariance head(unsigned L_t) const;
    IsoCovariance tail(unsigned L_w) const;
    mat get_mat() const override;
    vec get_vec() const;
    double get_scalar() const;
    unsigned get_size() const;
    double get() const { return get_scalar(); };

    // Useful for conversion std::vector<Covariance> <=> Armadillo
    using Type = vec;
    static vec getTypeSize(unsigned K, unsigned dimension) { return vec(K); };

    // Assignement operators
    IsoCovariance &operator=(const IsoCovariance &cov);
    IsoCovariance &operator=(const mat &cov);
    IsoCovariance &operator=(double scalar);
    IsoCovariance &operator+=(const mat &cov);
    IsoCovariance &operator+=(double scalar);

private:
    double scalar;
    double size;
};

// =================== Operator overload declarations ===================

/**
 * @brief Addition operator redefinition
 * @details The method performs C = A + B where C and A are armadillo matrices and B is of type Covariance.
 * @param y : armadillo::Mat<double>
 * @param x : Covariance
 * @return : armadillo::Mat<double>
 */
mat operator+(const mat &y, const FullCovariance &x);
mat operator+(const mat &y, const DiagCovariance &x);
mat operator+(const mat &y, const IsoCovariance &x);

/**
 * @brief Addition operator redefinition
 * @details The method performs C = A + B where C and B are armadillo matrices and A is of type Covariance.
 * @param x : Covariance
 * @param y : armadillo::Mat<double>
 * @return : armadillo::Mat<double>
 */
mat operator+(const FullCovariance &x, const mat &y);
mat operator+(const DiagCovariance &x, const mat &y);
mat operator+(const IsoCovariance &x, const mat &y);

/**
 * @brief Subtraction operator redefinition
 * @details The method performs C = A - B where C and A are armadillo matrices and B is of type Covariance.
 * @param y : armadillo::Mat<double>
 * @param x : Covariance
 * @return : armadillo::Mat<double>
 */
mat operator-(const mat &y, const FullCovariance &x);
mat operator-(const mat &y, const DiagCovariance &x);
mat operator-(const mat &y, const IsoCovariance &x);

/**
 * @brief Subtraction operator redefinition
 * @details The method performs C = A - B where C and B are armadillo matrices and A is of type Covariance.
 * @param x : Covariance
 * @param y : armadillo::Mat<double>
 * @return : armadillo::Mat<double>
 */
mat operator-(const FullCovariance &x, const mat &y);
mat operator-(const DiagCovariance &x, const mat &y);
mat operator-(const IsoCovariance &x, const mat &y);

/**
 * @brief Multiplication operator redefinition
 * @details The method performs C = A * B where C and A are armadillo matrices and B is of type Covariance.
 * @param y : armadillo::Mat<double>
 * @param x : Covariance
 * @return : armadillo::Mat<double>
 */
mat operator*(const mat &y, const FullCovariance &x);
mat operator*(const mat &y, const DiagCovariance &x);
mat operator*(const mat &y, const IsoCovariance &x);

/**
 * @brief Multiplication operator redefinition
 * @details The method performs C = A * B where C and A are armadillo subview_cols<double> and B is of type Covariance.
 * @param y : armadillo::Mat<double>
 * @param x : Covariance
 * @return : armadillo::Mat<double>
 */
mat operator*(const arma::subview_cols<double> &y, const FullCovariance &x);
mat operator*(const arma::subview_cols<double> &y, const DiagCovariance &x);
mat operator*(const arma::subview_cols<double> &y, const IsoCovariance &x);

/**
 * @brief Multiplication operator redefinition
 * @details The method performs C = A * B where C and B are armadillo matrices and A is of type Covariance.
 * @param x : Covariance
 * @param y : armadillo::Mat<double>
 * @return : armadillo::Mat<double>
 */
mat operator*(const FullCovariance &x, const mat &y);
mat operator*(const DiagCovariance &x, const mat &y);
mat operator*(const IsoCovariance &x, const mat &y);

/**
 * @brief Matrix-vector multiplication operator redefinition
 * @details The method performs C = A * B where C and B are column vectors and A is of type Covariance.
 * @param y : armadillo::Col<double>
 * @param x : Covariance
 * @return : armadillo::Col<double>
 */
vec operator*(const FullCovariance &x, const vec &y);
vec operator*(const DiagCovariance &x, const vec &y);
vec operator*(const IsoCovariance &x, const vec &y);

/**
 * @brief Vector-matrix multiplication operator redefinition
 * @details The method performs C = A * B where C and A are row vectors and A is of type Covariance.
 * @param y : armadillo::Row<double>
 * @param x : Covariance
 * @return : armadillo::Row<double>
 */
rowvec operator*(const rowvec &y, const FullCovariance &x);
rowvec operator*(const rowvec &y, const DiagCovariance &x);
rowvec operator*(const rowvec &y, const IsoCovariance &x);

#endif // COVARIANCE_HPP
