#ifndef XLLIM_TESTMODEL_H
#define XLLIM_TESTMODEL_H

#include "FunctionalModel.hpp"

/**
 * @class TestModel
 * @brief A class describing the Test model
 *
 * @details This class inherits @ref FunctionalModel "FunctionalModel" and overrides its methods by respecting the
 * equations in Test model.
 *
 */

class TestModel : public FunctionalModel
{
public:
    /**
     * @brief Constructor
     */
    TestModel();
    void F(vec x, vec &y) final;
    unsigned getDimensionY() final;
    unsigned getDimensionX() final;
    void toPhysic(vec &x) final;
    void fromPhysic(vec &x) final;

private:
    mat A_;
};

#endif // XLLIM_TESTMODEL_H
