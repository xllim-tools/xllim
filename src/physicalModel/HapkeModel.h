//
// Created by reverse-proxy on 18‏/12‏/2019.
//

#ifndef UNTITLED_HAPKEMODEL_H
#define UNTITLED_HAPKEMODEL_H

#define DEFAULT_B0 0.0
#define DEFAULT_H 0.1

#include <gtest/gtest_prod.h>

#include "FunctionnalModel.h"
#include "Enumeration.h"
#include <utility>

class HapkeModel : public FunctionnalModel{
public:

    HapkeModel(std::vector<std::vector<double>> &geometries);
    void F(const std::vector<double> &x, std::vector<double> &y) final ;
    std::vector<double> F(const std::vector<double> &x) final;
    std::vector<std::vector<double>> F(const std::vector<std::vector<double>> &x) final;
    int get_D_dimension() final;
    int get_L_dimension() final;
    std::vector<double> nomalize(std::vector<double> x) final;
    std::vector<double> invNormalize(std::vector<double> x) final;

protected:
    mat geom_helper_mat;
    mat configuredGeometries;
    double L_dimension = 6;

    FRIEND_TEST(Hapke02ModelTest, CalculateP);
    FRIEND_TEST(Hapke02ModelTest, ConfigureGeometries);

    void setupGeometries(mat geometries);
    rowvec calculate_P(double b, double c);
    rowvec calculate_S(double theta_bar, const rowvec& mue, const rowvec& mu0e,const rowvec& mue_0, const rowvec& mu0e_0);
    rowvec calculate_B(double b0, double h);
    rowvec calculate_MuE(double theta_bar, double E1_THETA_BAR, double E2_THETA_BAR);
    rowvec calculate_Mu0E(double theta_bar, double E1_THETA_BAR, double E2_THETA_BAR);
    rowvec calculate_MuE_0(double theta_bar, double E1_THETA_BAR, double E2_THETA_BAR);
    rowvec calculate_Mu0E_0(double theta_bar, double E1_THETA_BAR, double E2_THETA_BAR);
    static double calculate_X(double theta_bar);
    vec calculate_f(const vec &psi);

    double calculate_E1_THETA_BAR(double theta_bar);
    double calculate_E2_THETA_BAR(double theta_bar);

    virtual double set_coef() = 0;
    virtual rowvec define_different_part(const rowvec &x, rowvec mue, rowvec mu0e) = 0;

private:
    FRIEND_TEST(Hapke02ModelTest, CalculateX);
    static double degToGrad(double degree);
    static void calculate_phase_angle(const vec &theta, const vec &theta_0, const vec &psi, subview_col<double> g, subview_col<double> cos_g);
    static void calculate_alpha(const vec &theta_0, subview_col<double> alpha);
    void generate_geom_heper_mat();
    static void infinity_to_max(subview_col<double> x);
};


#endif //UNTITLED_HAPKEMODEL_H
