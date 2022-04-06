/**
 * @file HapkeModel.h
 * @brief Hapke model class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 18/12/2019
 */

#ifndef KERNELO_HAPKEMODEL_H
#define KERNELO_HAPKEMODEL_H

#include "HapkeAdapter.h"
#include "../FunctionalModel.h"
#include <utility>
#include <memory>

namespace Functional {

/**
 * @class HapkeModel
 * @brief Abstract class representing Hapke's model
 *
 * @details This class inherits @ref FunctionalModel "FunctionalModel" and contains a template
 * method for calculating F(x), where common parts are implemented in this class and different
 * parts are overloaded in the different versions of the model using subclasses.
 * The calculation in the class are initially designed for a 4 parameters model. The class can
 * use a @ref HapkeAdapter "HapkeAdapter" to adapt different variants of the model for example :
 * the model with 3 or 6 parameters.
 *
 * See : Hapke B. 1993 Theory of Reflectance and Emittance Spectroscopy. Topics in Remote Sensing.
 * Cambridge University Press, Cambridge, UK.
 *
 * See : Schmidt F. and Fernando J. 2015 Realistic uncertainties on Hapke model parameters from
 * photometric measurement. Icarus, 260 :73 - 93, 2015.
 *
 */
    class HapkeModel : public FunctionalModel {
    public:
        /**
         * @brief Constructor
         * @param geometries : pointer to the matrix of geometries that will be used by the model.
         * @param row_size : number of geometries.
         * @param col_size : number of parameters per geometry (should equals 3).
         * @param adapter : a shared pointer to the @ref HapkeAdapter "adapter".
         * @param theta_bar_scaling : value used to transform theta_bar between physical and mathematical spaces.
         */
        HapkeModel(const double *geometries, unsigned int row_size, unsigned int col_size, const std::shared_ptr<HapkeAdapter> &adapter,
                   double theta_bar_scaling);

        void F(rowvec photometry, rowvec &reflectances) final;

        int get_D_dimension() final;

        int get_L_dimension() final;

        void to_physic(rowvec &x) final;

        void to_physic(double *x, unsigned int size) final;

        void from_physic(double *x, unsigned int size) final;

    protected:
        mat geom_helper_mat; /**< A matrix containing intermediate results in relation to geometries */
        mat configuredGeometries; /**< A matrix of the configured geometries */
        std::shared_ptr<HapkeAdapter> adapter;/**< A shared pointer to the @ref HapkeAdapter "adapter" */
        double theta_bar_scaling; /**< A value used to transform theta_bar between physical and mathematical spaces */

        // FRIEND_TEST(Hapke02ModelTest, CalculateP);
        // FRIEND_TEST(Hapke02ModelTest, CalculateX);
        // FRIEND_TEST(Hapke02ModelTest, CalculateB);
        friend class Hapke02ModelTest_CalculateP_Test;
        friend class Hapke02ModelTest_CalculateX_Test;
        friend class Hapke02ModelTest_CalculateB_Test;

        /**
         * This method configures the geometries and prepares it for the calculation of reflectances
         * @param geometries : a matrix of 3 columns (theta, theta_0, psi)
         */
        void setupGeometries(mat geometries);

        /**
         * This method implements the particles scattering phase function
         * @param b : asymetry of the phase function
         * @param c : fraction of the backward scattering
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
        rowvec
        calculate_S(double theta_bar, const rowvec &mue, const rowvec &mu0e, const rowvec &mue_0, const rowvec &mu0e_0);

        /**
         * This method calculates the opposition effect
         * @param b0 : magnitude of the opposition effect
         * @param h : angular width of the opposition effect
         * @return a vector of D results
         */
        rowvec calculate_B(double b0, double h);

        /**
         * This method calculates MuE(THETA)
         * @param theta_bar : surface macroscopic roughness
         * @param E1 : E1 value dependent only on THETA_BAR
         * @param E1_0 : E2 value dependent only on THETA_BAR
         * @return a vector of D results
         */
        rowvec calculate_MuE(double theta_bar, rowvec &E1, rowvec &E1_0, rowvec &E2, rowvec &E2_0);

        /**
         * This method calculates Mu0E(THETA)
         * @param theta_bar : surface macroscopic roughness
         * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
         * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
         * @return a vector of D results
         */
        rowvec calculate_Mu0E(double theta_bar, rowvec &E1, rowvec &E1_0, rowvec &E2, rowvec &E2_0);

        /**
         * This method calculates MuE(0)
         * @param theta_bar : surface macroscopic roughness
         * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
         * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
         * @return a vector of D results
         */
        rowvec calculate_MuE_0(double theta_bar, rowvec &E1, rowvec &E1_0, rowvec &E2, rowvec &E2_0);

        /**
         * This method calculates Mu0E(0)
         * @param theta_bar : surface macroscopic roughness
         * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
         * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
         * @return a vector of D results
         */
        rowvec calculate_Mu0E_0(double theta_bar, rowvec &E1, rowvec &E1_0, rowvec &E2, rowvec &E2_0);

        /**
         * This method calculates X(THETA_BAR)
         * @param theta_bar : surface macroscopic roughness
         * @return X(THETA_BAR)
         */
        static double calculate_X(double theta_bar);

        /**
         * This method calculates f(PSI)
         * @param psi : azimuth
         * @return a vector of D results
         */
        static vec calculate_f(const vec &psi);

        /**
         * This is a virtual method returning the coefficient of the reflectance formula, it must
         * be overridden by subclasses
         * @return coefficient
         */
        virtual double set_coef() = 0;

        /**
         * This method is virtual, it represents the changing part in the reflectance formula.
         * Its implementation depends on Hapke's model version. It must be overridden in subclasses
         * @param photometry : a vector of a photometry parameters
         * @param mue
         * @param mu0e
         * @return a vector of D results
         */
        virtual rowvec define_different_part(const rowvec &photometry, rowvec mue, rowvec mu0e) = 0;

    private:

        /**
         * This method transforms a value in degree to a value in gradient.
         * @param degree
         * @return gradient value
         */
        static double degToGrad(double degree);

        /**
         * This method calculate the phase angle g and its cosinus and save it in the configured geometries.
         * @param theta : view zenith angle
         * @param theta_0 : solar zenith angle
         * @param psi : azimuth
         * @param g : phase angle
         * @param cos_g
         */
        static void calculate_phase_angle(const vec &theta, const vec &theta_0, const vec &psi, subview_col<double> g,
                                          subview_col<double> cos_g);

        /**
         * This method calculates alpha and save it in the configured geometries.
         * @param theta_0 : solar zenith angle
         * @param alpha
         */
        static void calculate_alpha(const vec &theta_0, subview_col<double> alpha);

        /**
         * This method generate the matrix of intermediate values used to speed up the calculus
         */
        void generate_geom_heper_mat();
    };

}

#endif //KERNELO_HAPKEMODEL_H
