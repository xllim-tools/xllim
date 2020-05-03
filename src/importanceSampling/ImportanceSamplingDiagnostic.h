//
// Created by reverse-proxy on 26‏/4‏/2020.
//

#ifndef KERNELO_IMPORTANCESAMPLINGDIAGNOSTIC_H
#define KERNELO_IMPORTANCESAMPLINGDIAGNOSTIC_H

namespace importanceSampling{
    struct ImportanceSamplingDiagnostic{
        unsigned nb_effective_sample;
        double effective_sample_size;
        double qn;
    };
}


#endif //KERNELO_IMPORTANCESAMPLINGDIAGNOSTIC_H
