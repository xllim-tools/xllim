//
// Created by sam on 02/05/22.
//

#ifndef KERNELO_TESTMODEL_H
#define KERNELO_TESTMODEL_H

#include "../FunctionalModel.h"


namespace Functional {
    /**
     * @class TestModel
     * @brief A class describing the Test model
     *
     * @details This class inherits @ref FunctionalModel "FunctionalModel" and overrides its methods by respecting the
     * equations in Test model.
     *
     */

    class TestModel : public FunctionalModel {
    public:
        /**
         * @brief Constructor
         */
        TestModel();
        void F(rowvec photometry, rowvec &reflectances) final;
        int get_D_dimension() final;
        int get_L_dimension() final;
        void to_physic(rowvec &x) final;
        void to_physic(double *x, unsigned int size) final;
        void from_physic(double *x, unsigned int size) final;

    private:
        mat A;

    };

}
#endif //KERNELO_TESTMODEL_H
