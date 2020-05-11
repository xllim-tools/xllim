/**
 * @file ShkuratovModel.cpp
 * @brief Shkuratov model class implementation
 * @author Sami DJOUADI
 * @version 1.1
 * @date 12/03/2020
 */

#include "ShkuratovModel.h"

#define DEGREE_180 180
#define L_dimension 5
#define INC 0
#define EME 1
#define PHI 2

using namespace Functional;
using namespace ShkuratovEnumeration;

ShkuratovModel::ShkuratovModel(const double *geometries, int row_size, int col_size,
                                           const double *scalingCoeffs, const double *offset) {

    this->scalingCoeffs = vec(&scalingCoeffs[0], L_dimension);
    this->offset = vec(&offset[0], L_dimension);

    mat geomsMat = mat(row_size,col_size);
    for(unsigned i=0; i<row_size; i++){
        for(unsigned j=0; j<col_size; j++){
            geomsMat(i,j) = geometries[i*col_size+j];
        }
    }
    setupGeometries(geomsMat);
}

void ShkuratovModel::F(rowvec photometry, rowvec &reflectances) {
    to_physic(photometry);

    vec cos_i = cos(configuredGeometries.col(BETA)) % cos(configuredGeometries.col(ShkuratovEnumeration::ALPHA) - configuredGeometries.col(GAMMA));
    vec f = (exp(- photometry(MU_1) * configuredGeometries.col(ShkuratovEnumeration::ALPHA)) + photometry(M) * exp(- photometry(MU_2) * configuredGeometries.col(ShkuratovEnumeration::ALPHA))) / (1 + photometry(M));
    vec d = cos(configuredGeometries.col(ShkuratovEnumeration::ALPHA) / 2.0) % cos(datum::pi * (configuredGeometries.col(GAMMA) - configuredGeometries.col(ShkuratovEnumeration::ALPHA) / 2.0) / (datum::pi - configuredGeometries.col(ShkuratovEnumeration::ALPHA))) / cos(configuredGeometries.col(GAMMA));
    for(unsigned i=0; i<d.n_rows; i++){
        d(i) *= pow(cos(configuredGeometries(i,BETA)), photometry(NU) * configuredGeometries(i,ShkuratovEnumeration::ALPHA) * (datum::pi - configuredGeometries(i,ShkuratovEnumeration::ALPHA)));
    }
    reflectances = photometry(AN) * d.t() % f.t() / cos_i.t();

}

int ShkuratovModel::get_D_dimension() {
    return configuredGeometries.n_rows;
}

int ShkuratovModel::get_L_dimension() {
    return L_dimension;
}

void ShkuratovModel::to_physic(rowvec &x) {
    for(unsigned l=0; l<x.n_cols; l++){
        x(l) = x(l) * scalingCoeffs(l) + offset(l);
    }
}

void ShkuratovModel::from_physic(double *x, int size) {
    for(unsigned l=0; l<size; l++){
        x[l] = (x[l] - offset(l)) / scalingCoeffs(l) ;
    }
}

void ShkuratovModel::setupGeometries(const mat &geometries) {
    configuredGeometries = mat(geometries.n_rows, geometries.n_cols, fill::zeros);
    mat geomsGrad = geometries;
    geomsGrad.transform( [](double val) {
        return degToGrad(val);
    });

    //compute Alpha
    configuredGeometries.col(ShkuratovEnumeration::ALPHA) = acos(cos(geomsGrad.col(INC)) % cos(geomsGrad.col(EME)) + sin(geomsGrad.col(INC)) % sin(geomsGrad.col(EME)) % cos(geomsGrad.col(PHI)));

    //compute Beta
    vec sin_i_e_2 = pow(sin(geomsGrad.col(INC) + geomsGrad.col(EME)),2);
    vec cos_phiDiv2_2 = pow(cos(geomsGrad.col(PHI)/2.0),2);
    vec sin_2_i = sin(geomsGrad.col(INC) * 2);
    vec sin_2_e = sin(geomsGrad.col(EME) * 2);
    vec cos_beta = sqrt(
            (sin_i_e_2 - cos_phiDiv2_2 % sin_2_i % sin_2_e) /
            (sin_i_e_2 - cos_phiDiv2_2 % sin_2_i % sin_2_e + pow(sin(geomsGrad.col(EME)),2) % pow(sin(geomsGrad.col(INC)),2) % pow(sin(geomsGrad.col(PHI)),2)));
    configuredGeometries.col(BETA) = acos(cos_beta);

    //compute Gamma
    configuredGeometries.col(GAMMA) = acos(cos(geomsGrad.col(EME)) / cos_beta);
}

double ShkuratovModel::degToGrad(double degree) {
    return degree * datum::pi / DEGREE_180;
}
