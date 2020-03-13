//
// Created by reverse-proxy on 12‏/3‏/2020.
//

#include "ShkuratovModel.h"

#define DEGREE_180 180

using namespace Functional;
using namespace ShkuratovEnumeration;

ShkuratovModel::ShkuratovModel(const double *geometries, int row_size, int col_size,
                                           const double *scalingCoeffs, const double *offset) {

    this->scalingCoeffs = vec(scalingCoeffs, 5);
    this->offset = vec(offset, 5);

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

    vec cos_i = cos(configuredGeometries.col(BETA)) % cos(configuredGeometries.col(ALPHA) - configuredGeometries.col(GAMMA));
    vec f = (exp(- photometry(MU_1) * configuredGeometries.col(ALPHA)) + photometry(M) * exp(- photometry(MU_2) * configuredGeometries.col(ALPHA))) / (1 + photometry(M));
    vec d = cos(configuredGeometries.col(ALPHA) / 2.0) % cos(datum::pi * (configuredGeometries.col(GAMMA) - configuredGeometries.col(ALPHA) / 2.0) / (datum::pi - configuredGeometries.col(ALPHA))) / cos(configuredGeometries.col(GAMMA));
    for(unsigned i=0; i<d.n_rows; i++){
        d(i) *= pow(cos(configuredGeometries(i,BETA)), photometry(NU) * configuredGeometries(i,ALPHA) * (datum::pi - configuredGeometries(i,ALPHA)));
    }
    reflectances = photometry(AN) * d.t() % f.t() / cos_i.t();

}

int ShkuratovModel::get_D_dimension() {
    return configuredGeometries.n_rows;
}

int ShkuratovModel::get_L_dimension() {
    return 5;
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
    configuredGeometries.col(ALPHA) = acos(cos(geomsGrad.col(0)) % cos(geomsGrad.col(1)) + sin(geomsGrad.col(0)) % sin(geomsGrad.col(1)) % cos(geomsGrad.col(2)));

    //compute Beta
    vec sin_i_e_2 = pow(sin(geomsGrad.col(0) + geomsGrad.col(1)),2);
    vec cos_phiDiv2_2 = pow(cos(geomsGrad.col(2)/2.0),2);
    vec sin_2_i = sin(geomsGrad.col(0) * 2);
    vec sin_2_e = sin(geomsGrad.col(1) * 2);
    vec cos_beta = sqrt(
            (sin_i_e_2 - cos_phiDiv2_2 % sin_2_i % sin_2_e) /
            (sin_i_e_2 - cos_phiDiv2_2 % sin_2_i % sin_2_e + pow(sin(geomsGrad.col(1)),2) % pow(sin(geomsGrad.col(0)),2) % pow(sin(geomsGrad.col(2)),2)));
    configuredGeometries.col(BETA) = acos(cos_beta);

    //compute Gamma
    configuredGeometries.col(GAMMA) = acos(cos(geomsGrad.col(1)) / cos_beta);
}

double ShkuratovModel::degToGrad(double degree) {
    return degree * datum::pi / DEGREE_180;
}
