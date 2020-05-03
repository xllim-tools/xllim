//
// Created by reverse-proxy on 4‏/4‏/2020.
//

#include "ISTargetDependent.h"
#include <utility>

using namespace importanceSampling;

ISTargetDependent::ISTargetDependent(std::shared_ptr<ISProposition> proposition) {
    this->proposition = std::move(proposition);
}

double ISTargetDependent::target_log_density(const vec &x, const vec &y, const vec &y_cov) {
    return ISTarget::target_log_density(x, y, y_cov); //+ this->proposition->proposition_log_density(x);
}
