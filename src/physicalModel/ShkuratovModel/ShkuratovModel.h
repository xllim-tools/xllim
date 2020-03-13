//
// Created by reverse-proxy on 12‏/3‏/2020.
//

#ifndef KERNELO_SHKURATOVMODEL_H
#define KERNELO_SHKURATOVMODEL_H

#include "../FunctionalModel.h"
#include "../Enumeration.h"

namespace Functional {
    class ShkuratovModel : public FunctionalModel {
    public:
        ShkuratovModel(const double *geometries, int row_size, int col_size, const double *scalingCoeffs, const double *offset);
        void F(rowvec photometry, rowvec &reflectances) final;
        int get_D_dimension() final;
        int get_L_dimension() final;
        void to_physic(rowvec &x) final;
        void from_physic(double *x, int size) final;

    protected:
        mat configuredGeometries;
        vec scalingCoeffs;
        vec offset;

    private:
        void setupGeometries(const mat &geometries);
        static double degToGrad(double degree);
    };

}


#endif //KERNELO_SHKURATOVMODEL_H
