#ifndef XLLIM_HAPKEMODEL_H
#define XLLIM_HAPKEMODEL_H

// #include "HapkeAdapter.h"
#include "FunctionalModel.hpp"
// #include <utility>
// #include <memory>

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
class HapkeModel : public FunctionalModel
{
public:
    /**
     * @brief Constructor
     * @param geometries : pointer to the matrix of geometries that will be used by the model.  Size should be (n_geometrie,3)
     * @param variant : a shared pointer to the @ref HapkeAdapter "adapter".
     * @param adapter : a shared pointer to the @ref HapkeAdapter "adapter".
     * @param theta_bar_scaling : value used to transform theta_bar between physical and mathematical spaces.
     */
    HapkeModel(mat geometries, std::string variant, std::string adapter, double theta_bar_scaling, double b0, double h);
    void F(vec photometry, vec &reflectances) final;
    unsigned getDimensionY() final;
    unsigned getDimensionX() final;
    void toPhysic(vec &x) final;
    void fromPhysic(vec &x) final;

protected:
    mat geom_helper_mat_;      /**< A matrix containing intermediate results in relation to geometries */
    mat configuredGeometries_; /**< A matrix of the configured geometries */
    std::string variant_;      /**< A shared pointer to the @ref HapkeAdapter "adapter" */
    std::string adapter_;      /**< A shared pointer to the @ref HapkeAdapter "adapter" */
    double theta_bar_scaling_; /**< A value used to transform theta_bar between physical and mathematical spaces */
    unsigned int L_dimension_; /** The dimension corresponds the the model variant */
    double b0_;
    double h_;
    double c_;

    // // FRIEND_TEST(Hapke02ModelTest, CalculateP);
    // // FRIEND_TEST(Hapke02ModelTest, CalculateX);
    // // FRIEND_TEST(Hapke02ModelTest, CalculateB);
    // friend class Hapke02ModelTest_CalculateP_Test;
    // friend class Hapke02ModelTest_CalculateX_Test;
    // friend class Hapke02ModelTest_CalculateB_Test;

    /**
     * This method adapt the model variable according dapter version
     * @param adapter : string adapter version among ["three", "four", "six"]
     */
    void adaptModel(vec &photometry);

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
    vec calculate_P(double b, double c);

    /**
     * This method calculates the macroscopic roughness factor
     * @param theta_bar : surface macroscopic roughness
     * @param mue
     * @param mu0e
     * @param mue_0
     * @param mu0e_0
     * @return a vector of D results
     */
    vec calculate_S(double theta_bar, const vec &mue, const vec &mu0e, const vec &mue_0, const vec &mu0e_0);

    /**
     * This method calculates the opposition effect
     * @param b0 : magnitude of the opposition effect
     * @param h : angular width of the opposition effect
     * @return a vector of D results
     */
    vec calculate_B(double b0, double h);

    /**
     * This method calculates MuE(THETA)
     * @param theta_bar : surface macroscopic roughness
     * @param E1 : E1 value dependent only on THETA_BAR
     * @param E1_0 : E2 value dependent only on THETA_BAR
     * @return a vector of D results
     */
    vec calculate_MuE(double theta_bar, vec &E1, vec &E1_0, vec &E2, vec &E2_0);

    /**
     * This method calculates Mu0E(THETA)
     * @param theta_bar : surface macroscopic roughness
     * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
     * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
     * @return a vector of D results
     */
    vec calculate_Mu0E(double theta_bar, vec &E1, vec &E1_0, vec &E2, vec &E2_0);

    /**
     * This method calculates MuE(0)
     * @param theta_bar : surface macroscopic roughness
     * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
     * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
     * @return a vector of D results
     */
    vec calculate_MuE_0(double theta_bar, vec &E1, vec &E1_0, vec &E2, vec &E2_0);

    /**
     * This method calculates Mu0E(0)
     * @param theta_bar : surface macroscopic roughness
     * @param E1_THETA_BAR : E1 value dependent only on THETA_BAR
     * @param E2_THETA_BAR : E2 value dependent only on THETA_BAR
     * @return a vector of D results
     */
    vec calculate_Mu0E_0(double theta_bar, vec &E1, vec &E1_0, vec &E2, vec &E2_0);

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
    double set_coef();

    /**
     * This method is virtual, it represents the changing part in the reflectance formula.
     * Its implementation depends on Hapke's model version. It must be overridden in subclasses
     * @param photometry : a vector of a photometry parameters
     * @param mue
     * @param mu0e
     * @return a vector of D results
     */
    // vec define_different_part(const vec &photometry, vec mue, vec mu0e);
    vec calculate_H(const vec &x, double omega);

private:
    static constexpr double DEGREE_180 = 180;
    enum photometry
    {
        OMEGA,     /**< single scattering albedo : index value 0*/
        THETA_BAR, /**< macroscopic roughness : index value 1*/
        B,         /**< asymetry of the phase function : index value 2*/
        C,         /**< fraction of the backward scattering : index value 3*/
        B0,        /**< amplitude of the opposition effect : index value 4*/
        H          /**< angular width of the opposition effect : index value 5*/
    };
    enum geometry
    {
        THETA_0 = 0, /**< view zenith angle, equivalent to INC : index value 0*/
        THETA = 1,   /**< solar zenith angle, equivalent to EME : index value 1*/
        PSI = 2,     /**< azimuth : index value 2*/
        ALPHA = 3,   /**< alpha : index value 3*/
        G = 4,       /**< phase angle : index value 4*/
        COS_G = 5    /**< cosinus of the phase angle : index value 5*/
    };
    enum geom_helper_index
    {
        COS_THETA = 0,
        SIN_THETA = 1,
        COS_THETA_0 = 2,
        SIN_THETA_0 = 3,
        SIN2_PSI_DIV2 = 4,
        TAN_G_DIV_2 = 5,
        F_PSI = 6,
        COS_PSI = 7,
        TAN_THETA = 8,
        TAN_THETA_0 = 9
    };

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
    void generate_geom_helper_mat();
};

#endif // XLLIM_HAPKEMODEL_H
