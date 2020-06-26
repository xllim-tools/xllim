/**
 * @file HapkeModel.cpp
 * @brief Hapke model class implementation
 * @author Sami DJOUADI
 * @version 1.0
 * @date 18/12/2019
 */

#include "../HapkeModel.h"
#include <utility>

#define DEGREE_180 180

using namespace Functional;
using namespace HapkeEnumeration;

// this index is used to access the matrix of intermediate results
enum geom_helper_index{
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

//-------------------------------- PUBLIC ------------------------------------//
HapkeModel::HapkeModel(const double *geometries, int row_size, int col_size,
                       const std::shared_ptr<HapkeAdapter> &adapter,
                       double theta_bar_scaling) {
    // Transform the geometry structure from double * to armadillo::mat
    mat geomsMat = mat(row_size,col_size);
    this->theta_bar_scaling = theta_bar_scaling;
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



    rowvec E1 = exp(-2 / datum::pi * geom_helper_mat.col(TAN_THETA) / tan(photometry(THETA_BAR))).t();
    rowvec E1_0 = exp(-2 / datum::pi * geom_helper_mat.col(TAN_THETA_0) / tan(photometry(THETA_BAR))).t();
    rowvec E2 = exp(-1 / datum::pi * pow(geom_helper_mat.col(TAN_THETA) / tan(photometry(THETA_BAR)),2)).t();
    rowvec E2_0 = exp(-1 / datum::pi * pow(geom_helper_mat.col(TAN_THETA_0) / tan(photometry(THETA_BAR)),2)).t();

    rowvec mu0e = calculate_Mu0E(photometry(THETA_BAR), E1, E1_0,E2, E2_0);
    rowvec mue = calculate_MuE(photometry(THETA_BAR), E1, E1_0,E2, E2_0);
    rowvec mue_0 = calculate_MuE_0(photometry(THETA_BAR), E1, E1_0,E2, E2_0);
    rowvec mu0e_0 = calculate_Mu0E_0(photometry(THETA_BAR), E1, E1_0,E2, E2_0);


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
    x[THETA_BAR] *= theta_bar_scaling;
    x[OMEGA] = 1 - pow(1- x[OMEGA], 2);
}

void HapkeModel::from_physic(double *x, int size) {
    // Denormalize THETA_BAR
    if(THETA_BAR < size){
        x[OMEGA] = 1 - sqrt(1 - x[OMEGA]);
        x[THETA_BAR] /= theta_bar_scaling;
    }
}

//--------------------------------------- PRIVATE METHODS ----------------------------------------//
void HapkeModel::generate_geom_heper_mat() {
    geom_helper_mat = mat(configuredGeometries.n_rows,10);
    geom_helper_mat.col(COS_THETA) = cos(configuredGeometries.col(THETA));
    geom_helper_mat.col(SIN_THETA) = sin(configuredGeometries.col(THETA));
    geom_helper_mat.col(COS_THETA_0) = cos(configuredGeometries.col(THETA_0));
    geom_helper_mat.col(SIN_THETA_0) = sin(configuredGeometries.col(THETA_0));
    geom_helper_mat.col(SIN2_PSI_DIV2) = pow(sin(configuredGeometries.col(PSI)/2),2);
    geom_helper_mat.col(TAN_G_DIV_2) = tan(configuredGeometries.col(G)/2);
    geom_helper_mat.col(F_PSI) = calculate_f(configuredGeometries.col(PSI));
    geom_helper_mat.col(COS_PSI) = cos(configuredGeometries.col(PSI));
    geom_helper_mat.col(TAN_THETA) = 1/ tan(configuredGeometries.col(THETA));
    geom_helper_mat.col(TAN_THETA_0) = 1/ tan(configuredGeometries.col(THETA_0));
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

//------------------------------------------- PROTECTED METHODS ---------------------------------------//

void HapkeModel::setupGeometries(mat geometries) {
    configuredGeometries = std::move(geometries);

    // transform degrees to gradients
    configuredGeometries.transform( [](double val) {
        return degToGrad(val);
    });

    configuredGeometries.resize(configuredGeometries.n_rows,6);

    // calculate phase angle and its cosinus
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
    rowvec result = b0 / (1 + geom_helper_mat.col(TAN_G_DIV_2).t() / h);
    return result;
}

rowvec HapkeModel::calculate_MuE(double theta_bar, rowvec &E1, rowvec &E1_0, rowvec &E2, rowvec &E2_0) {
    vec result = vec(configuredGeometries.n_rows);

    double tan_theta_bar = tan(theta_bar);

    for(unsigned i=0; i<configuredGeometries.n_rows; i++){
        if(configuredGeometries(i,THETA) >= configuredGeometries(i,THETA_0)){
            result(i) =
                    (
                            geom_helper_mat(i,COS_THETA) + geom_helper_mat(i,SIN_THETA) * tan_theta_bar *
                            (E2(i) - geom_helper_mat(i,SIN2_PSI_DIV2) * E2_0(i)) /
                            (2 - E1(i) - configuredGeometries(i,PSI) / datum::pi * E1_0(i))
                    );
        }else{
            result(i) =
                    (
                            geom_helper_mat(i,COS_THETA) + geom_helper_mat(i,SIN_THETA) * tan_theta_bar *
                            (geom_helper_mat(i,COS_PSI) * E2_0(i) + geom_helper_mat(i,SIN2_PSI_DIV2) * E2(i)) /
                            (2 - E1_0(i) - configuredGeometries(i,PSI) / datum::pi * E1(i))
                    );
        }
    }
    return result.t() * calculate_X(theta_bar);
}

rowvec HapkeModel::calculate_Mu0E(double theta_bar, rowvec &E1, rowvec &E1_0, rowvec &E2, rowvec &E2_0) {
    vec result = vec(configuredGeometries.n_rows);

    double tan_theta_bar = tan(theta_bar);

    for(unsigned i=0; i<configuredGeometries.n_rows; i++){
        if(configuredGeometries(i,THETA) >= configuredGeometries(i,THETA_0)){
            result(i) =(
                                geom_helper_mat(i,COS_THETA_0) + geom_helper_mat(i,SIN_THETA_0) * tan_theta_bar *
                                (geom_helper_mat(i,COS_PSI) * E2(i) + geom_helper_mat(i,SIN2_PSI_DIV2) * E2_0(i)) /
                                (2 - E1(i) - configuredGeometries(i,PSI) / datum::pi * E1_0(i))
                        );
        }else{
            result(i) =(
                                geom_helper_mat(i,COS_THETA_0) + geom_helper_mat(i,SIN_THETA_0) * tan_theta_bar *
                                (E2_0(i) - geom_helper_mat(i,SIN2_PSI_DIV2) * E2(i)) /
                                (2 - E1_0(i) - configuredGeometries(i,PSI) / datum::pi * E1(i))
                        );
        }
    }
    return result.t() * calculate_X(theta_bar);
}

double HapkeModel::calculate_X(const double theta_bar) {
    return 1 / sqrt(1 + datum::pi * pow(tan(theta_bar), 2));
}

rowvec HapkeModel::calculate_MuE_0(double theta_bar, rowvec &E1, rowvec &E1_0, rowvec &E2, rowvec &E2_0) {
    vec result = calculate_X(theta_bar) *
            (geom_helper_mat.col(COS_THETA) +
            (geom_helper_mat.col(SIN_THETA) % E2.t() *tan(theta_bar) /(2 - E1.t())));
    return result.t();
}

rowvec HapkeModel::calculate_Mu0E_0(double theta_bar, rowvec &E1, rowvec &E1_0, rowvec &E2, rowvec &E2_0) {
    vec result = calculate_X(theta_bar) *
              (geom_helper_mat.col(COS_THETA_0) +
              (geom_helper_mat.col(SIN_THETA_0) % E2_0.t() * tan(theta_bar) /(2 - E1_0.t()))
              );
    return result.t();
}











