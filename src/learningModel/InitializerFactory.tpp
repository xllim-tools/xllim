/**
 * @file InitializerFactory.tpp
 * @author Sami DJOUADI
 * @version 1.1
 * @date 19/03/2020
 */

#include "configs/InitConfig.h"
#include "initializers/FixedInitializer.h"
#include "initializers/MultInitializer.h"

using namespace learningModel;

template<typename T, typename U>
std::shared_ptr<Iinitilizer<T, U>> InitializerFactory::create(const std::shared_ptr<InitConfig> &initConfig) {
    static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
    static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    if(std::dynamic_pointer_cast<FixedInitConfig>(initConfig)){
        return std::make_shared<FixedInitializer<T,U>>(
                FixedInitializer<T,U>(std::dynamic_pointer_cast<FixedInitConfig>(initConfig))
        );
    }else{
        return std::make_shared<MultInitializer<T,U>>(
                MultInitializer<T,U>(std::dynamic_pointer_cast<MultInitConfig>(initConfig))
        );
    }
}