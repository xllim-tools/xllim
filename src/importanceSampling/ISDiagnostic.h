//
// Created by reverse-proxy on 2‏/4‏/2020.
//

#ifndef KERNELO_ISDIAGNOSTIC_H
#define KERNELO_ISDIAGNOSTIC_H

namespace importanceSampling{
    struct ISDiagnostic{
        int nb_effective_sample;
        double effective_sample_size;
        double qn;
    };
}


#endif //KERNELO_ISDIAGNOSTIC_H
