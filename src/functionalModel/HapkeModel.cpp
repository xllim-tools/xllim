#include "HapkeModel.hpp"
#include <stdexcept>  // for std::runtime_error
// #include "../../src/logging/Logger.h"
// #include <utility>

// #define DEGREE_180 180

// using namespace HapkeEnumeration;
// using namespace Logging;

//-------------------------------- PUBLIC ------------------------------------//

HapkeModel::HapkeModel(mat geometries, std::string variant, std::string adapter, double theta_bar_scaling, double b0, double h)
{
    variant_ = variant;
    adapter_ = adapter;
    b0_ = b0;
    h_ = h;
    theta_bar_scaling_ = theta_bar_scaling;
    if (adapter_ == "three")
    {
        L_dimension_ = 3;
    }
    else if (adapter_ == "four")
    {
        L_dimension_ = 4;
    }
    else if (adapter_ == "six")
    {
        L_dimension_ = 6;
    }
    else
    {
        // Logging::Logger::GetInstance()->log("\tInvalid Hapke adapter version", Logging::Logger::level(Logging::ERROR));
    }
    setupGeometries(geometries);
}

void HapkeModel::F(vec photometry, vec &reflectances)
{
    toPhysic(photometry);                                     // Transform photometry from mathematical space to physical space
    photometry(THETA_BAR) = degToGrad(photometry(THETA_BAR)); // Set THETA_BAR to radian
    adaptModel(photometry);                                   // Adapting Hapke model according to adapter type

    vec E1 = exp(-2 / datum::pi * geom_helper_mat_.col(TAN_THETA) / tan(photometry(THETA_BAR)));
    vec E1_0 = exp(-2 / datum::pi * geom_helper_mat_.col(TAN_THETA_0) / tan(photometry(THETA_BAR)));
    vec E2 = exp(-1 / datum::pi * pow(geom_helper_mat_.col(TAN_THETA) / tan(photometry(THETA_BAR)), 2));
    vec E2_0 = exp(-1 / datum::pi * pow(geom_helper_mat_.col(TAN_THETA_0) / tan(photometry(THETA_BAR)), 2));
    vec mu0e = calculate_Mu0E(photometry(THETA_BAR), E1, E1_0, E2, E2_0);
    vec mue = calculate_MuE(photometry(THETA_BAR), E1, E1_0, E2, E2_0);
    vec mue_0 = calculate_MuE_0(photometry(THETA_BAR), E1, E1_0, E2, E2_0);
    vec mu0e_0 = calculate_Mu0E_0(photometry(THETA_BAR), E1, E1_0, E2, E2_0);

    // Caculate reflectances
    vec specific_part = (1 + calculate_B(b0_, h_)) % calculate_P(photometry(B), c_) + calculate_H(mu0e, photometry(OMEGA)) % calculate_H(mue, photometry(OMEGA)) - 1;

    reflectances = set_coef() * (photometry(OMEGA) / configuredGeometries_.col(ALPHA) % mu0e / (mue + mu0e)) % specific_part % calculate_S(photometry(THETA_BAR), mue, mu0e, mue_0, mu0e_0);
}

unsigned HapkeModel::getDimensionY()
{
    return configuredGeometries_.n_rows;
}

unsigned HapkeModel::getDimensionX()
{
    return L_dimension_;
}

void HapkeModel::toPhysic(vec &x)
{
    x(OMEGA) = 1 - pow(1 - x(OMEGA), 2);
    x(THETA_BAR) *= theta_bar_scaling_;
}

void HapkeModel::fromPhysic(vec &x)
{
    x(OMEGA) = 1 - sqrt(1 - x(OMEGA));
    x(THETA_BAR) /= theta_bar_scaling_;
}

//--------------------------------------- PRIVATE METHODS ----------------------------------------//
void HapkeModel::generate_geom_helper_mat()
{
    geom_helper_mat_ = mat(configuredGeometries_.n_rows, 10);
    geom_helper_mat_.col(COS_THETA) = cos(configuredGeometries_.col(THETA));
    geom_helper_mat_.col(SIN_THETA) = sin(configuredGeometries_.col(THETA));
    geom_helper_mat_.col(COS_THETA_0) = cos(configuredGeometries_.col(THETA_0));
    geom_helper_mat_.col(SIN_THETA_0) = sin(configuredGeometries_.col(THETA_0));
    geom_helper_mat_.col(SIN2_PSI_DIV2) = pow(sin(configuredGeometries_.col(PSI) / 2), 2);
    geom_helper_mat_.col(TAN_G_DIV_2) = tan(configuredGeometries_.col(G) / 2);
    geom_helper_mat_.col(F_PSI) = calculate_f(configuredGeometries_.col(PSI));
    geom_helper_mat_.col(COS_PSI) = cos(configuredGeometries_.col(PSI));
    geom_helper_mat_.col(TAN_THETA) = 1 / tan(configuredGeometries_.col(THETA));
    geom_helper_mat_.col(TAN_THETA_0) = 1 / tan(configuredGeometries_.col(THETA_0));
}

double HapkeModel::degToGrad(double degree)
{
    return degree * datum::pi / DEGREE_180;
}

void HapkeModel::calculate_phase_angle(const vec &theta, const vec &theta_0, const vec &psi, subview_col<double> g, subview_col<double> cos_g)
{
    cos_g = cos(theta_0) % cos(theta) + sin(theta) % sin(theta_0) % cos(psi);
    g = acos(cos_g);
}

void HapkeModel::calculate_alpha(const vec &theta_0, subview_col<double> alpha)
{
    alpha = 4 * cos(theta_0);
}

//------------------------------------------- PROTECTED METHODS ---------------------------------------//

void HapkeModel::adaptModel(vec &photometry)
{
    if (adapter_ == "three")
    {
        // b0_ = b0;
        // h_ = h;
        c_ = (3.29 * exp(-17.4 * pow(photometry(B), 2)) + 0.092) / 2;
        // L_dimension_ = 3;
    }
    else if (adapter_ == "four")
    {
        // b0_ = b0;
        // h_ = h;
        c_ = photometry(C);
        // L_dimension_ = 4;
    }
    else if (adapter_ == "six")
    {
        b0_ = photometry(B0);
        h_ = photometry(H);
        c_ = photometry(C);
        // L_dimension_ = 6;
    }
    else
    {
        // Logging::Logger::GetInstance()->log("\tInvalid Hapke adapter version", Logging::Logger::level(Logging::ERROR));
    }
}

void HapkeModel::setupGeometries(mat geometries)
{
    configuredGeometries_ = std::move(geometries);
    configuredGeometries_.transform([](double val)
                                   { return degToGrad(val); });  // transform degrees to gradients
    configuredGeometries_.resize(configuredGeometries_.n_rows, 6); // adding columns for ALPHA, G and COS_G
    calculate_phase_angle(
        configuredGeometries_.col(THETA),
        configuredGeometries_.col(THETA_0),
        configuredGeometries_.col(PSI),
        configuredGeometries_.col(G),
        configuredGeometries_.col(COS_G));
    calculate_alpha(
        configuredGeometries_.col(THETA_0),
        configuredGeometries_.col(ALPHA));
    generate_geom_helper_mat();
}

double HapkeModel::set_coef()
{
    if (variant_ == "1993")
    {
        return 1;
    }
    else if (variant_ == "2002")
    {
        return 1;
    }
    else
    {
        // Logging::Logger::GetInstance()->log("\tInvalid Hapke model version", Logging::Logger::level(Logging::ERROR));
        throw std::runtime_error("Invalid Hapke model version");  // throw by value, not pointer (S1035)
    }
}

vec HapkeModel::calculate_H(const vec &x, double omega)
{
    if (variant_ == "1993")
    {
        double y = sqrt(1 - omega);
        double temp = (1 - y) / (1 + y);
        vec result = 1 / (1 - (1 - y) * x % (temp + (1 - (0.5 + x) * temp) % log((1 + x) / x)));
        return result;
    }
    else if (variant_ == "2002")
    {
        double y = sqrt(1 - omega);
        vec result = (1 + 2 * x) / (1 + 2 * x * y);
        return result;
    }
    else
    {
        // Logging::Logger::GetInstance()->log("\tInvalid Hapke model version", Logging::Logger::level(Logging::ERROR));
        throw std::runtime_error("Invalid Hapke model version");  // throw by value, not pointer (S1035)
    }
}

vec HapkeModel::calculate_f(const vec &psi)
{
    return exp(-2 * tan(psi / 2));
}

vec HapkeModel::calculate_P(const double b, const double c)
{
    vec P = vec(configuredGeometries_.n_rows);
    double b2 = pow(b, 2);
    for (unsigned i = 0; i < configuredGeometries_.n_rows; i++)
    {
        P(i) = (1 - b2) *
               ((1 - c) / pow(1 + 2 * b * configuredGeometries_(i, COS_G) + b2, 1.5) +
                c / pow(1 - 2 * b * configuredGeometries_(i, COS_G) + b2, 1.5));
    }
    return P;
}

vec HapkeModel::calculate_S(const double theta_bar, const vec &mue, const vec &mu0e, const vec &mue_0, const vec &mu0e_0)
{
    vec result = vec(configuredGeometries_.n_rows);
    double x_theta_bar = calculate_X(theta_bar);

    for (unsigned i = 0; i < configuredGeometries_.n_rows; i++)
    {
        if (configuredGeometries_(i, THETA) >= configuredGeometries_(i, THETA_0))
        {
            result(i) = geom_helper_mat_(i, COS_THETA_0) * mue(i) / mue_0(i) / mu0e_0(i) * x_theta_bar /
                        (1 - geom_helper_mat_(i, F_PSI) + geom_helper_mat_(i, F_PSI) * x_theta_bar * (geom_helper_mat_(i, COS_THETA_0) / mu0e_0(i)));
        }
        else
        {
            result(i) = geom_helper_mat_(i, COS_THETA_0) * mue(i) / mue_0(i) / mu0e_0(i) * x_theta_bar /
                        (1 - geom_helper_mat_(i, F_PSI) + geom_helper_mat_(i, F_PSI) * x_theta_bar * (geom_helper_mat_(i, COS_THETA) / mue_0(i)));
        }
    }
    return result;
}

vec HapkeModel::calculate_B(const double b0, const double h)
{
    vec result = b0 / (1 + geom_helper_mat_.col(TAN_G_DIV_2) / h);
    return result;
}

vec HapkeModel::calculate_MuE(double theta_bar, vec &E1, vec &E1_0, vec &E2, vec &E2_0)
{
    vec result = vec(configuredGeometries_.n_rows);

    double tan_theta_bar = tan(theta_bar);

    for (unsigned i = 0; i < configuredGeometries_.n_rows; i++)
    {
        if (configuredGeometries_(i, THETA) >= configuredGeometries_(i, THETA_0))
        {
            result(i) =
                (geom_helper_mat_(i, COS_THETA) + geom_helper_mat_(i, SIN_THETA) * tan_theta_bar *
                                                     (E2(i) - geom_helper_mat_(i, SIN2_PSI_DIV2) * E2_0(i)) /
                                                     (2 - E1(i) - configuredGeometries_(i, PSI) / datum::pi * E1_0(i)));
        }
        else
        {
            result(i) =
                (geom_helper_mat_(i, COS_THETA) + geom_helper_mat_(i, SIN_THETA) * tan_theta_bar *
                                                     (geom_helper_mat_(i, COS_PSI) * E2_0(i) + geom_helper_mat_(i, SIN2_PSI_DIV2) * E2(i)) /
                                                     (2 - E1_0(i) - configuredGeometries_(i, PSI) / datum::pi * E1(i)));
        }
    }
    return result * calculate_X(theta_bar);
}

vec HapkeModel::calculate_Mu0E(double theta_bar, vec &E1, vec &E1_0, vec &E2, vec &E2_0)
{
    vec result = vec(configuredGeometries_.n_rows);

    double tan_theta_bar = tan(theta_bar);

    for (unsigned i = 0; i < configuredGeometries_.n_rows; i++)
    {
        if (configuredGeometries_(i, THETA) >= configuredGeometries_(i, THETA_0))
        {
            result(i) = (geom_helper_mat_(i, COS_THETA_0) + geom_helper_mat_(i, SIN_THETA_0) * tan_theta_bar *
                                                               (geom_helper_mat_(i, COS_PSI) * E2(i) + geom_helper_mat_(i, SIN2_PSI_DIV2) * E2_0(i)) /
                                                               (2 - E1(i) - configuredGeometries_(i, PSI) / datum::pi * E1_0(i)));
        }
        else
        {
            result(i) = (geom_helper_mat_(i, COS_THETA_0) + geom_helper_mat_(i, SIN_THETA_0) * tan_theta_bar *
                                                               (E2_0(i) - geom_helper_mat_(i, SIN2_PSI_DIV2) * E2(i)) /
                                                               (2 - E1_0(i) - configuredGeometries_(i, PSI) / datum::pi * E1(i)));
        }
    }
    return result * calculate_X(theta_bar);
}

double HapkeModel::calculate_X(const double theta_bar)
{
    return 1 / sqrt(1 + datum::pi * pow(tan(theta_bar), 2));
}

vec HapkeModel::calculate_MuE_0(double theta_bar, vec &E1, vec &E1_0, vec &E2, vec &E2_0)
{
    vec result = calculate_X(theta_bar) *
                 (geom_helper_mat_.col(COS_THETA) +
                  (geom_helper_mat_.col(SIN_THETA) % E2 * tan(theta_bar) / (2 - E1)));
    return result;
}

vec HapkeModel::calculate_Mu0E_0(double theta_bar, vec &E1, vec &E1_0, vec &E2, vec &E2_0)
{
    vec result = calculate_X(theta_bar) *
                 (geom_helper_mat_.col(COS_THETA_0) +
                  (geom_helper_mat_.col(SIN_THETA_0) % E2_0 * tan(theta_bar) / (2 - E1_0)));
    return result;
}
