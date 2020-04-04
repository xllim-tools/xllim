//
// Created by reverse-proxy on 4‏/4‏/2020.
//

#ifndef KERNELO_ISTARGETDEPENDENT_H
#define KERNELO_ISTARGETDEPENDENT_H

#include "ISTarget.h"
#include "ISProposition.h"

namespace importanceSampling{
    class ISTargetDependent : public ISTarget{
    public:
        explicit ISTargetDependent(std::shared_ptr<ISProposition> proposition);
        double target_log_density(const vec &x, const vec &y, const vec &y_cov) override;

    private:
        std::shared_ptr<ISProposition> proposition;
    };

}



#endif //KERNELO_ISTARGETDEPENDENT_H
