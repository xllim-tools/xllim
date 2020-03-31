//
// Created by reverse-proxy on 19‏/3‏/2020.
//

#include "gllim/GLLiMLearning.h"
#include "EstimatorFactory.h"
#include "InitializerFactory.h"
#include "estimators/GmmEstimator.h"

using namespace learningModel;

std::shared_ptr<IGLLiMLearning> LearningModelFactory::create(
        unsigned k,
        const std::string &GammaType,
        const std::string &SigmaType,
        const std::shared_ptr<InitConfig>& initConfig,
        const std::shared_ptr<LearningConfig>& learningConfig) {

    if(GammaType == "Full"){
        if(SigmaType == "Full"){ // special case where the estimator is a GMM estimator
            return std::make_shared<GLLiMLearning<FullCovariance, FullCovariance>>(
                    GLLiMLearning<FullCovariance, FullCovariance>(
                            InitializerFactory::create<FullCovariance, FullCovariance>(initConfig),
                            std::make_shared<GmmEstimator>(
                                    GmmEstimator(std::dynamic_pointer_cast<GMMLearningConfig>(learningConfig))
                            ),
                            k));
        }else if(SigmaType == "Diag"){
            return std::make_shared<GLLiMLearning<FullCovariance, DiagCovariance>>(
                    GLLiMLearning<FullCovariance, DiagCovariance>(
                            InitializerFactory::create<FullCovariance, DiagCovariance>(initConfig),
                            EstimatorFactory::create<FullCovariance,DiagCovariance>(learningConfig),
                            k));
        }else if(SigmaType == "Iso"){
            return std::make_shared<GLLiMLearning<FullCovariance, IsoCovariance>>(
                    GLLiMLearning<FullCovariance, IsoCovariance>(
                            InitializerFactory::create<FullCovariance, IsoCovariance>(initConfig),
                            EstimatorFactory::create<FullCovariance,IsoCovariance>(learningConfig),
                            k));
        }
    }else if(GammaType == "Diag"){
        if(SigmaType == "Full"){
            return std::make_shared<GLLiMLearning<DiagCovariance, FullCovariance>>(
                    GLLiMLearning<DiagCovariance, FullCovariance>(
                            InitializerFactory::create<DiagCovariance, FullCovariance>(initConfig),
                            EstimatorFactory::create<DiagCovariance,FullCovariance>(learningConfig),
                            k));
        }else if(SigmaType == "Diag"){
            return std::make_shared<GLLiMLearning<DiagCovariance, DiagCovariance>>(
                    GLLiMLearning<DiagCovariance, DiagCovariance>(
                            InitializerFactory::create<DiagCovariance, DiagCovariance>(initConfig),
                            EstimatorFactory::create<DiagCovariance,DiagCovariance>(learningConfig),
                            k));
        }else if(SigmaType == "Iso"){
            return std::make_shared<GLLiMLearning<DiagCovariance, IsoCovariance>>(
                    GLLiMLearning<DiagCovariance, IsoCovariance>(
                            InitializerFactory::create<DiagCovariance, IsoCovariance>(initConfig),
                            EstimatorFactory::create<DiagCovariance,IsoCovariance>(learningConfig),
                            k));
        }
    }else if(GammaType == "Iso"){
        if(SigmaType == "Full"){
            return std::make_shared<GLLiMLearning<IsoCovariance, FullCovariance>>(
                    GLLiMLearning<IsoCovariance, FullCovariance>(
                            InitializerFactory::create<IsoCovariance, FullCovariance>(initConfig),
                            EstimatorFactory::create<IsoCovariance,FullCovariance>(learningConfig),
                            k));
        }else if(SigmaType == "Diag"){
            return std::make_shared<GLLiMLearning<IsoCovariance, DiagCovariance>>(
                    GLLiMLearning<IsoCovariance, DiagCovariance>(
                            InitializerFactory::create<IsoCovariance, DiagCovariance>(initConfig),
                            EstimatorFactory::create<IsoCovariance,DiagCovariance>(learningConfig),
                            k));
        }else if(SigmaType == "Iso"){
            return std::make_shared<GLLiMLearning<IsoCovariance, IsoCovariance>>(
                    GLLiMLearning<IsoCovariance, IsoCovariance>(
                            InitializerFactory::create<IsoCovariance, IsoCovariance>(initConfig),
                            EstimatorFactory::create<IsoCovariance,IsoCovariance>(learningConfig),
                            k));
        }
    }
}
