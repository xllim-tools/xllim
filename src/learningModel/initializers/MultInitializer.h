//
// Created by reverse-proxy on 3‏/3‏/2020.
//

#ifndef KERNELO_MULTINITIALIZER_H
#define KERNELO_MULTINITIALIZER_H

#include "Initializers.h"
#include "../configs/InitConfig.h"
#include "../../dataGeneration/GeneratorStrategy.h"
#include <memory>

namespace learningModel{
    template <typename T , typename U >
    class MultInitializer : public Iinitilizer<T,U> {

    public:
        explicit MultInitializer(const std::shared_ptr<MultInitConfig>& config);
        std::shared_ptr<GLLiMParameters <T, U>> execute(const mat &x, const mat &y, unsigned K) override;

    private:
        std::shared_ptr<MultInitConfig> config;
    };
}

#include "MultInitializer.tpp"


#endif //KERNELO_MULTINITIALIZER_H
