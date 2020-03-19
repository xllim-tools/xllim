//
// Created by reverse-proxy on 3‏/3‏/2020.
//

#ifndef KERNELO_FIXEDINITIALIZER_H
#define KERNELO_FIXEDINITIALIZER_H

#include "Initializers.h"
#include "../configs/InitConfig.h"
#include "../../dataGeneration/GeneratorStrategy.h"
#include <memory>

namespace learningModel{

    template <typename T , typename U >
    class FixedInitializer : public Iinitilizer<T,U> {

    public:
        explicit FixedInitializer(const std::shared_ptr<FixedInitConfig>& config);
        std::shared_ptr<GLLiMParameters<T, U>> execute(const mat &x, const mat &y, unsigned K) override ;


    private:
        std::shared_ptr<FixedInitConfig> config;

    };
}

#include "FixedInitializer.tpp"


#endif //KERNELO_FIXEDINITIALIZER_H
