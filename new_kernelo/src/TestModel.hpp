#ifndef KERNELO_TESTMODEL_H
#define KERNELO_TESTMODEL_H

#include "FunctionalModel.hpp"


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
        void from_physic(rowvec &x) final;

    private:
        mat A;

    };

}
#endif //KERNELO_TESTMODEL_H
