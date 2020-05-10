/**
 * @file ISDiagnostic.h
 * @author Sami DJOUADI
 * @version 1.2
 * @date 22/04/2020
 */

#ifndef KERNELO_ISDIAGNOSTIC_H
#define KERNELO_ISDIAGNOSTIC_H

namespace importanceSampling{
    /**
     * @struct ISDiagnostic
     * @brief This struct wraps the results of the diagnostic of the importance sampling
     */
    struct ISDiagnostic{
        int nb_effective_sample;
        double effective_sample_size;
        double qn;
    };
}


#endif //KERNELO_ISDIAGNOSTIC_H
