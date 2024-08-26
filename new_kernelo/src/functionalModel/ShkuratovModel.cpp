#include "ShkuratovModel.hpp"

ShkuratovModel::ShkuratovModel(mat geometries, std::string variant, vec scalingCoeffs, vec offset)
{
    if (variant == "5p")
    {
        L_dimension_ = 5;
    }
    else if (variant == "3p")
    {
        L_dimension_ = 3;
    }
    scalingCoeffs_ = scalingCoeffs;
    offset_ = offset;
    setupGeometries(geometries);
    cos_i_ = cos(geometries.col(INC) * datum::pi / DEGREE_180);
}

void ShkuratovModel::F(vec photometry, vec &reflectances)
{
    toPhysic(photometry);

    vec f;
    if (L_dimension_ == 5)
    {
        f = (exp(-photometry(MU_1) * configuredGeometries_.col(ALPHA)) + photometry(M) * exp(-photometry(MU_2) * configuredGeometries_.col(ALPHA))) / (1 + photometry(M));
    }
    else if (L_dimension_ == 3)
    {
        f = exp(-photometry(MU_1) * configuredGeometries_.col(ALPHA));
    }
    vec d = cos(configuredGeometries_.col(ALPHA) / 2.0) % cos(datum::pi * (configuredGeometries_.col(GAMMA) - configuredGeometries_.col(ALPHA) / 2.0) / (datum::pi - configuredGeometries_.col(ALPHA))) / cos(configuredGeometries_.col(GAMMA));
    for (unsigned i = 0; i < d.n_rows; i++)
    {
        d(i) *= pow(cos(configuredGeometries_(i, BETA)), photometry(NU) * configuredGeometries_(i, ALPHA) * (datum::pi - configuredGeometries_(i, ALPHA)));
    }
    reflectances = photometry(AN) * d % f / cos_i_;
}

unsigned ShkuratovModel::getDimensionY()
{
    return configuredGeometries_.n_rows;
}

unsigned ShkuratovModel::getDimensionX()
{
    return L_dimension_;
}

void ShkuratovModel::toPhysic(vec &x)
{
    x = x % scalingCoeffs_ + offset_;
}

void ShkuratovModel::fromPhysic(vec &x)
{
    x = (x - offset_) / scalingCoeffs_;
}

void ShkuratovModel::setupGeometries(const mat &geometries)
{
    configuredGeometries_ = mat(geometries.n_rows, geometries.n_cols, fill::zeros);
    mat geomsGrad = geometries;
    geomsGrad.transform([](double val)
                        { return degToGrad(val); });

    // compute Alpha
    configuredGeometries_.col(ALPHA) = acos(cos(geomsGrad.col(INC)) % cos(geomsGrad.col(EME)) + sin(geomsGrad.col(INC)) % sin(geomsGrad.col(EME)) % cos(geomsGrad.col(PHI)));

    // compute Beta
    vec sin_i_e_2 = pow(sin(geomsGrad.col(INC) + geomsGrad.col(EME)), 2);
    vec cos_phiDiv2_2 = pow(cos(geomsGrad.col(PHI) / 2.0), 2);
    vec sin_2_i = sin(geomsGrad.col(INC) * 2);
    vec sin_2_e = sin(geomsGrad.col(EME) * 2);
    vec cos_beta = sqrt(
        (sin_i_e_2 - cos_phiDiv2_2 % sin_2_i % sin_2_e) /
        (sin_i_e_2 - cos_phiDiv2_2 % sin_2_i % sin_2_e + pow(sin(geomsGrad.col(EME)), 2) % pow(sin(geomsGrad.col(INC)), 2) % pow(sin(geomsGrad.col(PHI)), 2)));
    configuredGeometries_.col(BETA) = acos(cos_beta);

    // compute Gamma
    configuredGeometries_.col(GAMMA) = atan((cos(geomsGrad.col(INC)) / cos(geomsGrad.col(EME)) - cos(configuredGeometries_.col(ALPHA))) / sin(configuredGeometries_.col(ALPHA)));
}

double ShkuratovModel::degToGrad(double degree)
{
    return degree * datum::pi / DEGREE_180;
}
