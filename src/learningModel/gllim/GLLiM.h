//
// Created by reverse-proxy on 4‏/3‏/2020.
//

#ifndef KERNELO_GLLIM_H
#define KERNELO_GLLIM_H

namespace learningModel{

    class GLLiM {

    public:
        unsigned K;
        unsigned L;
        unsigned D;
        double *Pi; // K
        double *C; //L*K
        double *B; //D*K
        double *Gamma; //L*L*K
        double *Sigma; //D*D*K
        double *A; //D*L*K
    };

}

#endif //KERNELO_GLLIM_H
