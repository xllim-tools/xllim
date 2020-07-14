/**
 * @file creators.h
 * @brief Functional model module configuration structures
 * @author Sami DJOUADI
 * @version 1.2
 * @date 27/01/2020
 */

#ifndef KERNELO_DATAGENCREATORS_H
#define KERNELO_CREATORS_H

#include <string>
#include <memory>
#include "HapkeModel/HapkeAdapter.h"
#include "HapkeModel/HapkeVersions/Hapke02Model.h"
#include "HapkeModel/HapkeAdapters/SixParamsModel.h"
#include "HapkeModel/HapkeAdapters/FourParamsModel.h"
#include "HapkeModel/HapkeAdapters/ThreeParamsModel.h"
#include "HapkeModel/HapkeVersions/Hapke93Model.h"
#include "ShkuratovModel/ShkuratovModel.h"
#include "ExternalModel/ExternalFunctionalModel.h"

namespace Functional{

    /**
     * @struct HapkeAdapterConfig
     *
     * This struct wraps the parameters used to configure an adapter of the Hapke model. It contains the method
     * create that returns a shared pointer to a @ref HapkeAdapter "HapkeAdapter" object.
     */
    class HapkeAdapterConfig{
    public:
        std::string version; /**< A string that determines which adapter to create : six , four or three
 * parameters adapter. */
        double b0; /**< a default value to set on the photometric parameter B0 of the model in case a three or four
 * parameters adapter is required. */
        double h; /**< a default value to set on the photometric parameter H of the model in case a three or four
 * parameters adapter is required. */

        /**
         * This method creates an adapter of the Hapke model given the configuration parameters and returns a shared
         * pointer of it.
         *
         * @return A shared pointer of the created Hapke adapter object.
         */
        std::shared_ptr<HapkeAdapter> create(){
            if(version == "six") {
                return std::shared_ptr<HapkeAdapter>(new SixParamsModel());
            }else if(version == "four"){
                return std::shared_ptr<HapkeAdapter>(new FourParamsModel(this->b0, this->h));
            }else if(version == "three"){
                return std::shared_ptr<HapkeAdapter>(new ThreeParamsModel(this->b0, this->h));
            }
        }
    };

    /**
     * @struct HapkeModelConfig
     *
     * This struct wraps the parameters used to configure the Hapke model. It contains the method
     * create that returns a shared pointer to a @ref FunctionalModel "FunctionalModel" object.
     */
    struct HapkeModelConfig{
        std::string version; /**< A string that determines which version of the Hapke model is required. It may be
 * 2000 or 1993.*/
        std::shared_ptr<HapkeAdapterConfig> adapterConfig; /**< See documentation of @ref HapkeAdapterConfig "HapkeAdapterConfig" */
        const double *geometries; /**< A pointer to a matrix of geometries required to initialize a Hapke model. */
        int row_size; /**< The number of geometries. */
        int col_size; /**< The dimension of the geometries. */
        double theta_bar_scalling; /**< A value used to transform theta_bar between physical and mathematical spaces. */

        /**
         * This method creates an Hapke model object given the configuration parameters and returns a shared
         * pointer of it.
         *
         * @return A shared pointer of the created Hapke model object as @ref FunctionalModel "FunctionalModel"
         */
        std::shared_ptr<FunctionalModel> create(){
            std::shared_ptr<HapkeAdapter> adapter = this->adapterConfig->create();
            if(version == "2002"){
                return std::shared_ptr<FunctionalModel>(
                        new Hapke02Model(
                                geometries,
                                row_size,
                                col_size,
                                adapter,
                                theta_bar_scalling)
                                );
            }
            else if(version == "1993"){
                return std::shared_ptr<FunctionalModel>(
                        new Hapke93Model(
                                geometries,
                                row_size,
                                col_size,
                                adapter,
                                theta_bar_scalling)
                                );
            }
        }
    };

    /**
     * @struct ShkuratovModelConfig
     *
     * This struct wraps the parameters used to configure the Shkuratov model. It contains the method
     * create that returns a shared pointer to a @ref FunctionalModel "FunctionalModel" object.
     */
    struct ShkuratovModelConfig{
        const double *geometries; /**< A pointer to a matrix of geometries required to initialize a Hapke model. */
        int row_size; /**< The number of geometries. */
        int col_size; /**< The dimension of the geometries. */
        const double *scalingCoeffs; /**< A set of coefficients used in the transformation between physical and
 * mathematical spaces. */
        const double *offset; /**< Offsets used in the transformation between physical and
 * mathematical spaces. */

        /**
         * This method creates a Shkuratov model object given the configuration parameters and returns a shared
         * pointer of it.
         *
         * @return A shared pointer of the created Shkuratov model object as @ref FunctionalModel "FunctionalModel"
         */
        std::shared_ptr<FunctionalModel> create(){
            return std::shared_ptr<FunctionalModel>(
                    new ShkuratovModel(
                            geometries,
                            row_size,
                            col_size,
                            scalingCoeffs,
                            offset)
                    );
        }
    };

    /**
     * @struct ExternalModelConfig
     *
     * This struct wraps the parameters used to configure a functional model with a source code witten in python.
     * It contains the method create that returns a shared pointer to a @ref FunctionalModel "FunctionalModel" object.
     */
    struct ExternalModelConfig{
        std::string className; /**< The name of the concrete class that defines the required external model. */
        std::string fileName; /**< The name of the file where the source code of the external model is written. */
        std::string filePath; /**< The path to the file where the source code of the external model is written.*/

        /**
         * This method creates an @ref ExternalFunctionalModel "ExternalFunctionalModel" object given the configuration
         * parameters and returns a shared pointer of it.
         *
         * @return A shared pointer of the created Shkuratov model object as @ref FunctionalModel "FunctionalModel"
         */
        std::shared_ptr<FunctionalModel> create(){
            return std::shared_ptr<FunctionalModel>(
                    new ExternalFunctionalModel(
                            className,
                            fileName,
                            filePath)
                    );
        }
    };
}

#endif //KERNELO_DATAGENCREATORS_H
