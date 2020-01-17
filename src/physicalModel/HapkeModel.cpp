#include "HapkeModel.h"
#include <utility>
#include "HapkeAdapterFactory.h"

#define DEGREE_180 180
#define THETA_BAR_SCALING 30

using namespace HapkeEnumeration;

enum geom_helper_index{
    COS_THETA = 0,
    SIN_THETA = 1,
    COS_THETA_0 = 2,
    SIN_THETA_0 = 3,
    SIN2_PSI_DIV2 = 4,
    E1_THETA = 5,
    E1_THETA_0 = 6,
    E2_THETA = 7,
    E2_THETA_0 = 8,
    TAN_G_DIV_2 = 9,
    F_PSI = 10,
    COS_PSI = 11
};

//-------------------------------- PUBLIC ------------------------------------//
HapkeModel::HapkeModel(const double *geometries, int row_size, int col_size, const std::shared_ptr<HapkeAdapter>& adapter) {
    // Transform the geometry structure from double * to armadillo::mat
    mat geomsMat = mat(row_size,col_size);

    this->adapter = adapter;

    for(unsigned i=0; i<row_size; i++){
        for(unsigned j=0; j<col_size; j++){
            geomsMat(i,j) = geometries[i*col_size+j];
        }
    }

    //call setup geometries method
    setupGeometries(geomsMat);
}

void HapkeModel::F(rowvec photometry, rowvec &reflectances) {
    // transform photometry from mathematical space to physical space
    to_physic(photometry);

    //Set THETA_BAR to radian
    photometry(THETA_BAR) = degToGrad(photometry(THETA_BAR));

    //Adapting Hapke model
    adapter->adaptModel(photometry);

    double E1_THETA_BAR = calculate_E1_THETA_BAR(photometry(THETA_BAR));
    double E2_THETA_BAR = calculate_E2_THETA_BAR(photometry(THETA_BAR));

    rowvec mu0e = calculate_Mu0E(photometry(THETA_BAR), E1_THETA_BAR, E2_THETA_BAR);
    rowvec mue = calculate_MuE(photometry(THETA_BAR), E1_THETA_BAR, E2_THETA_BAR);
    rowvec mue_0 = calculate_MuE_0(photometry(THETA_BAR), E1_THETA_BAR, E2_THETA_BAR);
    rowvec mu0e_0 = calculate_Mu0E_0(photometry(THETA_BAR), E1_THETA_BAR, E2_THETA_BAR);

    //Caculate reflectances
    reflectances = set_coef()
            * (photometry(OMEGA) / configuredGeometries.col(ALPHA).t() % mu0e / (mue + mu0e))
            % define_different_part(photometry,mue, mu0e)
            % calculate_S(photometry(THETA_BAR), mue, mu0e, mue_0, mu0e_0);
}

int HapkeModel::get_D_dimension() {
    return configuredGeometries.n_rows;
}

int HapkeModel::get_L_dimension() {
    return adapter->get_dimension_L();
}

void HapkeModel::to_physic(rowvec &x) {
    // Normalize THETA_BAR
    x[THETA_BAR] *= THETA_BAR_SCALING;
}

void HapkeModel::from_physic(double *x, int size) {
    // Denormalize THETA_BAR
    x[THETA_BAR] /= THETA_BAR_SCALING;
}

//--------------------------------------- PRIVATE ----------------------------------------//

void HapkeModel::generate_geom_heper_mat() {
    geom_helper_mat = mat(configuredGeometries.n_rows,12);

    geom_helper_mat.col(COS_THETA) = cos(configuredGeometries.col(THETA));
    geom_helper_mat.col(SIN_THETA) = sin(configuredGeometries.col(THETA));
    geom_helper_mat.col(COS_THETA_0) = cos(configuredGeometries.col(THETA_0));
    geom_helper_mat.col(SIN_THETA_0) = sin(configuredGeometries.col(THETA_0));
    geom_helper_mat.col(SIN2_PSI_DIV2) = pow(sin(configuredGeometries.col(PSI)/2),2);

    geom_helper_mat.col(E1_THETA) = exp(-2 / datum::pi * geom_helper_mat.col(COS_THETA)/geom_helper_mat.col(SIN_THETA));
    geom_helper_mat.col(E1_THETA_0) = exp(-2 / datum::pi * geom_helper_mat.col(COS_THETA_0)/geom_helper_mat.col(SIN_THETA_0));
    geom_helper_mat.col(E2_THETA) = exp(-1 / datum::pi * pow(geom_helper_mat.col(COS_THETA)/geom_helper_mat.col(SIN_THETA),2));
    geom_helper_mat.col(E2_THETA_0) = exp(-1 / datum::pi * pow(geom_helper_mat.col(COS_THETA_0)/geom_helper_mat.col(SIN_THETA_0),2));

    geom_helper_mat.col(TAN_G_DIV_2) = tan(configuredGeometries.col(G)/2);
    geom_helper_mat.col(F_PSI) = calculate_f(configuredGeometries.col(PSI));
    geom_helper_mat.col(COS_PSI) = cos(configuredGeometries.col(PSI));

}

double HapkeModel::degToGrad(double degree) {
    return degree * datum::pi / DEGREE_180;
}

void HapkeModel::calculate_phase_angle(const vec &theta, const vec &theta_0, const vec &psi, subview_col<double> g, subview_col<double> cos_g){
    cos_g = cos(theta_0) % cos(theta) + sin(theta) % sin(theta_0) % cos(psi);
    g = acos(cos_g);
}

void HapkeModel::calculate_alpha(const vec &theta_0, subview_col<double> alpha) {
    alpha = 4 * cos(theta_0);
}

//------------------------------------------- PROTECTED ---------------------------------------//

void HapkeModel::setupGeometries(mat geometries) {
    configuredGeometries = std::move(geometries);

    // transform degrees to gradients
    configuredGeometries.transform( [](double val) {
        return degToGrad(val);
    });

    configuredGeometries.resize(configuredGeometries.n_rows,6);

    // calculate phase angle and his cosinus
    calculate_phase_angle(
            configuredGeometries.col(THETA),
            configuredGeometries.col(THETA_0),
            configuredGeometries.col(PSI),
            configuredGeometries.col(G),
            configuredGeometries.col(COS_G));

    // calculate alpha
    calculate_alpha(
            configuredGeometries.col(THETA_0),
            configuredGeometries.col(ALPHA));

    generate_geom_heper_mat();
}


vec HapkeModel::calculate_f(const vec &psi) {
    return exp(- 2 * tan(psi / 2));
}

rowvec HapkeModel::calculate_P(const double b, const double c) {
    vec P = vec(configuredGeometries.n_rows);
    double b2 = pow(b,2);
    for(unsigned i=0; i<configuredGeometries.n_rows; i++){
        P(i) = (1 - b2) *
                (
                        (1 - c) / pow(1 + 2 * b * configuredGeometries(i,COS_G) + b2,1.5)
                        +
                        c / pow(1 - 2 * b * configuredGeometries(i, COS_G) + b2,1.5)
                );
    }
    return P.t();
}

rowvec HapkeModel::calculate_S(const double theta_bar, const rowvec& mue, const rowvec& mu0e,const rowvec& mue_0, const rowvec& mu0e_0) {
    vec result = vec(configuredGeometries.n_rows);
    double x_theta_bar = calculate_X(theta_bar);

    for(unsigned i=0; i<configuredGeometries.n_rows; i++){
        if(configuredGeometries(i,THETA) >= configuredGeometries(i,THETA_0)){
            result(i) = geom_helper_mat(i,COS_THETA_0) * mue(i) / mue_0(i) / mu0e_0(i) * x_theta_bar /
                    ( 1 - geom_helper_mat(i,F_PSI) + geom_helper_mat(i,F_PSI) * x_theta_bar * (geom_helper_mat(i,COS_THETA_0) / mu0e_0(i)));
        }else{
            result(i) = geom_helper_mat(i,COS_THETA_0) * mue(i) / mue_0(i) / mu0e_0(i) * x_theta_bar /
                    ( 1 - geom_helper_mat(i,F_PSI) + geom_helper_mat(i,F_PSI) * x_theta_bar * ( geom_helper_mat(i,COS_THETA)/ mue_0(i)));
        }
    }
    return result.t();
}

rowvec HapkeModel::calculate_B(const double b0, const double h) {
    rowvec B = rowvec(configuredGeometries.n_rows);
    B = b0 / (1 + geom_helper_mat.col(TAN_G_DIV_2).t() / h);
    return B;
}

rowvec HapkeModel::calculate_MuE(const double theta_bar, const double E1_THETA_BAR, const double E2_THETA_BAR) {
    vec result = vec(configuredGeometries.n_rows);

    double tan_theta_bar = tan(theta_bar);

    for(unsigned i=0; i<configuredGeometries.n_rows; i++){
        if(configuredGeometries(i,THETA) >= configuredGeometries(i,THETA_0)){
            result(i) = calculate_X(theta_bar) *
                    (
                            geom_helper_mat(i,COS_THETA) + geom_helper_mat(i,SIN_THETA) * tan_theta_bar *
                            (pow(geom_helper_mat(i,E2_THETA),E2_THETA_BAR) - geom_helper_mat(i,SIN2_PSI_DIV2) * pow(geom_helper_mat(i,E2_THETA_0),E2_THETA_BAR)) /
                            (2 - pow(geom_helper_mat(i,E1_THETA),E1_THETA_BAR) - configuredGeometries(i,PSI) / datum::pi * pow(geom_helper_mat(i,E1_THETA_0),E1_THETA_BAR))
                    );
        }else{
            result(i) = calculate_X(theta_bar) *
                    (
                            geom_helper_mat(i,COS_THETA) + geom_helper_mat(i,SIN_THETA) * tan_theta_bar *
                            (geom_helper_mat(i,COS_PSI) * pow(geom_helper_mat(i,E2_THETA_0),E2_THETA_BAR) + geom_helper_mat(i,SIN2_PSI_DIV2) * pow(geom_helper_mat(i,E2_THETA),E2_THETA_BAR)) /
                            (2 - pow(geom_helper_mat(i,E1_THETA_0),E1_THETA_BAR) - configuredGeometries(i,PSI) / datum::pi * pow(geom_helper_mat(i,E1_THETA),E1_THETA_BAR))
                    );
        }
    }
    return result.t();
}

rowvec HapkeModel::calculate_Mu0E(const double theta_bar, const double E1_THETA_BAR, const double E2_THETA_BAR) {
    vec result = vec(configuredGeometries.n_rows);

    double tan_theta_bar = tan(theta_bar);

    for(unsigned i=0; i<configuredGeometries.n_rows; i++){
        if(configuredGeometries(i,THETA) >= configuredGeometries(i,THETA_0)){
            result(i) = calculate_X(theta_bar) *
                        (
                                geom_helper_mat(i,COS_THETA_0) + geom_helper_mat(i,SIN_THETA_0) * tan_theta_bar *
                                (geom_helper_mat(i,COS_PSI) * pow(geom_helper_mat(i,E2_THETA),E2_THETA_BAR) + geom_helper_mat(i,SIN2_PSI_DIV2) * pow(geom_helper_mat(i,E2_THETA_0),E2_THETA_BAR)) /
                                (2 - pow(geom_helper_mat(i,E1_THETA),E1_THETA_BAR) - configuredGeometries(i,PSI) / datum::pi * pow(geom_helper_mat(i,E1_THETA_0),E1_THETA_BAR))
                        );
        }else{
            result(i) = calculate_X(theta_bar) *
                        (
                                geom_helper_mat(i,COS_THETA_0) + geom_helper_mat(i,SIN_THETA_0) * tan_theta_bar *
                                (pow(geom_helper_mat(i,E2_THETA_0),E2_THETA_BAR) - geom_helper_mat(i,SIN2_PSI_DIV2) * pow(geom_helper_mat(i,E2_THETA),E2_THETA_BAR)) /
                                (2 - pow(geom_helper_mat(i,E1_THETA_0),E1_THETA_BAR) - configuredGeometries(i,PSI) / datum::pi * pow(geom_helper_mat(i,E1_THETA),E1_THETA_BAR))
                        );
        }
    }
    return result.t();
}

double HapkeModel::calculate_X(const double theta_bar) {
    return 1 / sqrt(1 + datum::pi * pow(tan(theta_bar), 2));
}

rowvec HapkeModel::calculate_MuE_0(const double theta_bar, const double E1_THETA_BAR, const double E2_THETA_BAR) {
    vec result = vec(configuredGeometries.n_rows);
    result  = calculate_X(theta_bar) *
            (geom_helper_mat.col(COS_THETA) +
                (
                        (geom_helper_mat.col(SIN_THETA) % pow(geom_helper_mat.col(E2_THETA),E2_THETA_BAR) *
                        tan(theta_bar) /(2 - pow(geom_helper_mat.col(E1_THETA), E1_THETA_BAR)))
                )

            );
    return result.t();
}

rowvec HapkeModel::calculate_Mu0E_0(const double theta_bar, const double E1_THETA_BAR, const double E2_THETA_BAR) {
    vec result = vec(configuredGeometries.n_rows);
    result  = calculate_X(theta_bar) *
              (geom_helper_mat.col(COS_THETA_0) +
                      (
                              (geom_helper_mat.col(SIN_THETA_0) % pow(geom_helper_mat.col(E2_THETA_0),E2_THETA_BAR) *
                              tan(theta_bar) /(2 - pow(geom_helper_mat.col(E1_THETA_0), E1_THETA_BAR)))
                      )
              );
    return result.t();
}

double HapkeModel::calculate_E1_THETA_BAR(const double theta_bar) {
    return std::min(std::numeric_limits<double>::max(), 1/tan(theta_bar));
}

double HapkeModel::calculate_E2_THETA_BAR(const double theta_bar) {
    return std::min(std::numeric_limits<double>::max(),  1/pow(tan(theta_bar),2));
}

void HapkeModel::infinity_to_max(subview_col<double> x) {
    for(unsigned i=0 ; i<x.n_rows ; i++){
        x(i) = std::min(std::numeric_limits<double>::max(), x(i));
    }
}











