/**
 * @file ImportanceSamplingDiagnostic.h
 * @author Sami DJOUADI
 * @version 1.2
 * @date 26/04/2020
 */

#ifndef KERNELO_IMPORTANCESAMPLINGDIAGNOSTIC_H
#define KERNELO_IMPORTANCESAMPLINGDIAGNOSTIC_H

namespace importanceSampling{
    /**
     * @struct ImportanceSamplingDiagnostic
     * @details This struct wraps the results of the diagnostic of the importance sampling. The struct is meant for integration purposes
     * with a third language API.
     */
    class ImportanceSamplingDiagnostic{
    public:
        unsigned nb_effective_sample;
        double effective_sample_size;
        double qn;
    };
}


#endif //KERNELO_IMPORTANCESAMPLINGDIAGNOSTIC_H
