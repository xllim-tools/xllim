#include "ShkuratovModel.hpp"

using namespace Functional;

ShkuratovModel::ShkuratovModel(mat geometries, std::string variant, vec scalingCoeffs, vec offset)
{
    if (variant == "5p")
    {
        this->L_dimension = 5;
    }
    else if (variant == "3p")
    {
        this->L_dimension = 3;
    }
    this->scalingCoeffs = scalingCoeffs;
    this->offset = offset;
    setupGeometries(geometries);
    this->cos_i = cos(geometries.col(INC) * datum::pi / DEGREE_180);
}

void ShkuratovModel::F(rowvec photometry, rowvec &reflectances)
{
    to_physic(photometry);

    vec f;
    if (this->L_dimension == 5)
    {
        f = (exp(-photometry(MU_1) * configuredGeometries.col(ALPHA)) + photometry(M) * exp(-photometry(MU_2) * configuredGeometries.col(ALPHA))) / (1 + photometry(M));
    }
    else if (this->L_dimension == 3)
    {
        f = exp(-photometry(MU_1) * configuredGeometries.col(ALPHA));
    }
    vec d = cos(configuredGeometries.col(ALPHA) / 2.0) % cos(datum::pi * (configuredGeometries.col(GAMMA) - configuredGeometries.col(ALPHA) / 2.0) / (datum::pi - configuredGeometries.col(ALPHA))) / cos(configuredGeometries.col(GAMMA));
    for (unsigned i = 0; i < d.n_rows; i++)
    {
        d(i) *= pow(cos(configuredGeometries(i, BETA)), photometry(NU) * configuredGeometries(i, ALPHA) * (datum::pi - configuredGeometries(i, ALPHA)));
    }
    reflectances = photometry(AN) * d.t() % f.t() / cos_i.t();
}

int ShkuratovModel::get_D_dimension()
{
    return configuredGeometries.n_rows;
}

int ShkuratovModel::get_L_dimension()
{
    return L_dimension;
}

void ShkuratovModel::to_physic(rowvec &x)
{
    x = x % scalingCoeffs + offset;
}

void ShkuratovModel::from_physic(rowvec &x)
{
    x = (x - offset) / scalingCoeffs;
}

void ShkuratovModel::setupGeometries(const mat &geometries)
{
    configuredGeometries = mat(geometries.n_rows, geometries.n_cols, fill::zeros);
    mat geomsGrad = geometries;
    geomsGrad.transform([](double val)
                        { return degToGrad(val); });

    // compute Alpha
    configuredGeometries.col(ALPHA) = acos(cos(geomsGrad.col(INC)) % cos(geomsGrad.col(EME)) + sin(geomsGrad.col(INC)) % sin(geomsGrad.col(EME)) % cos(geomsGrad.col(PHI)));

    // compute Beta
    vec sin_i_e_2 = pow(sin(geomsGrad.col(INC) + geomsGrad.col(EME)), 2);
    vec cos_phiDiv2_2 = pow(cos(geomsGrad.col(PHI) / 2.0), 2);
    vec sin_2_i = sin(geomsGrad.col(INC) * 2);
    vec sin_2_e = sin(geomsGrad.col(EME) * 2);
    vec cos_beta = sqrt(
        (sin_i_e_2 - cos_phiDiv2_2 % sin_2_i % sin_2_e) /
        (sin_i_e_2 - cos_phiDiv2_2 % sin_2_i % sin_2_e + pow(sin(geomsGrad.col(EME)), 2) % pow(sin(geomsGrad.col(INC)), 2) % pow(sin(geomsGrad.col(PHI)), 2)));
    configuredGeometries.col(BETA) = acos(cos_beta);

    // compute Gamma
    configuredGeometries.col(GAMMA) = atan((cos(geomsGrad.col(INC)) / cos(geomsGrad.col(EME)) - cos(configuredGeometries.col(ALPHA))) / sin(configuredGeometries.col(ALPHA)));
}

double ShkuratovModel::degToGrad(double degree)
{
    return degree * datum::pi / DEGREE_180;
}
