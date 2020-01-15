/**
 * @file HapkeModel.h
 * @brief Hapke model class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 18/12/2019
 */

#ifndef UNTITLED_HAPKEMODEL_H
#define UNTITLED_HAPKEMODEL_H


#include <gtest/gtest_prod.h>
#include "HapkeAdapter.h"
#include "FunctionnalModel.h"
#include "Enumeration.h"
#include <utility>
#include <memory>

/**
 * @class HapkeModel
 * @brief Abstract class representing the Hapke model
 *
 * @details This class contains a template method for calculating F(x). Where common
 * parts are implemented in this class and different parts are overloaded in
 * the different versions of the model using subclasses.
 *
 */
class HapkeModel : public FunctionnalModel{
public:
    /**
     * @brief Constructor
     * @details Hapke Model class constructor
     * @param geometries : matrix of geometries that will be used by the model
     */
    HapkeModel(const double *geometries, int row_size, int col_size, const std::shared_ptr<HapkeAdapter>& adapter);
    void F(const rowvec &x, rowvec &y) final ;
    int get_D_dimension() final;
    int get_L_dimension() final;
    void to_physic(double *x, int size) final;
    void from_physic(double *x, int size) final;

protected:
    mat geom_helper_mat; /**< A matrix containing intermediate results in relation to geometries */
    mat configuredGeometries; /**< A matrix of the configured geometries */
    std::shared_ptr<HapkeAdapter> adapter;

    FRIEND_TEST(Hapke02ModelTest, CalculateP);
    FRIEND_TEST(Hapke02ModelTest, CalculateB);
    //FRIEND_TEST(Hapke02ModelTest, ConfigureGeometries);

    /**
     * This method configures the geometries and prepare it for the calculation of reflectances
     * @param geometries : a matrix of 3 columns (theta, theta_0, psi)
     */
    void setupGeometries(mat geometries);

    /**
     * This method implements the particles scattering phase function
     * @param b asymmetry parameter
     * @param c backscattering fraction
     * @return a vector of D results
     */
    rowvec calculate_P(double b, double c);

    /**
     * This method calculates the macroscopic roughness factor
     * @param theta_bar : surface macroscopic roughness
     * @param mue
     * @param mu0e
     * @param mue_0
     * @param mu0e_0
     * @return a vector of D results
     */
    rowvec calculate_S(double theta_bar, const rowvec& mue, const rowvec& mu0e,const rowvec& mue_0, const rowvec& mu0e_0);

    /**
     * This method calculates the opposition effect
     * @param b0 : magnitude of the opposition effect
     * @param h : angular width of the opposition effect
     * @return a vector of D results
     */
    rowvec calculate_B(double b0, double h);

    /**
     * This method calculated MuE(THETA)
     * @param theta_bar : surface macroscopic roughness
     * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
     * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
     * @return a vector of D results
     */
    rowvec calculate_MuE(double theta_bar, double E1_THETA_BAR, double E2_THETA_BAR);

    /**
     * This method calculated Mu0E(THETA)
     * @param theta_bar : surface macroscopic roughness
     * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
     * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
     * @return a vector of D results
     */
    rowvec calculate_Mu0E(double theta_bar, double E1_THETA_BAR, double E2_THETA_BAR);

    /**
     * This method calculated MuE(0)
     * @param theta_bar : surface macroscopic roughness
     * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
     * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
     * @return a vector of D results
     */
    rowvec calculate_MuE_0(double theta_bar, double E1_THETA_BAR, double E2_THETA_BAR);

    /**
     * This method calculated Mu0E(0)
     * @param theta_bar : surface macroscopic roughness
     * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
     * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
     * @return a vector of D results
     */
    rowvec calculate_Mu0E_0(double theta_bar, double E1_THETA_BAR, double E2_THETA_BAR);

    /**
     * This method calculated X(THETA_BAR)
     * @param theta_bar : surface macroscopic roughness
     * @return X(THETA_BAR)
     */
    static double calculate_X(double theta_bar);

    /**
     * This method calculates f(PSI)
     * @param psi
     * @return a vector of D results
     */
    vec calculate_f(const vec &psi);


    double calculate_E1_THETA_BAR(double theta_bar);
    double calculate_E2_THETA_BAR(double theta_bar);

    /**
     * This is a virtual class returning the coefficient of the reflectance formula, it must
     * be overridden by subclasses
     * @return coefficient
     */
    virtual double set_coef() = 0;

    /**
     * This method is virtual, it represents the changing part in the reflectance formula.
     * Its implementation depends on the Hapke model version. It must be overridden in subclasses
     * @param x : a vector of a photometry parameters
     * @param mue
     * @param mu0e
     * @return a vector of D results
     */
    virtual rowvec define_different_part(const rowvec &x, rowvec mue, rowvec mu0e) = 0;

private:
    FRIEND_TEST(Hapke02ModelTest, CalculateX);

    /**
     * This method transforms a value in degree to a value in gradient.
     * @param degree
     * @return gradient value
     */
    static double degToGrad(double degree);

    /**
     * This method calculate the phase angle g and its cosinus and save it in the configured geometries.
     * @param theta
     * @param theta_0
     * @param psi
     * @param g
     * @param cos_g
     */
    static void calculate_phase_angle(const vec &theta, const vec &theta_0, const vec &psi, subview_col<double> g, subview_col<double> cos_g);

    /**
     * This method calculates alpha and save it in the configured geometries.
     * @param theta_0
     * @param alpha
     */
    static void calculate_alpha(const vec &theta_0, subview_col<double> alpha);

    /**
     * This method generate the matrix of intermediate values used to speed up the calculus
     */
    void generate_geom_heper_mat();
    static void infinity_to_max(subview_col<double> x);
};


#endif //UNTITLED_HAPKEMODEL_H
