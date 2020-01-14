//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_GENERATORSTRATEGY_H
#define KERNELO_GENERATORSTRATEGY_H

namespace DataGeneration{
    class GeneratorStrategy{
    public:
        virtual void execute(int n, int dimension, mat &x) = 0;
    };
}

#endif //KERNELO_GENERATORSTRATEGY_H