/**
 * @file Enumeration.h
 * @brief This file contains different enumerations used in the functional model
 * @author Sami DJOUADI
 * @version 1.0
 * @date 18/12/2019
 */

namespace Functional {
/**
 * @namespace HapkeEnumeration
 * Enumerations used in Hapke's model
 */
    namespace HapkeEnumeration {

        /**
         * @enum photometry data index enumeration
         */
        enum photometry {
            OMEGA, /**< single scattering albedo : index value 0*/
            THETA_BAR, /**< macroscopic roughness : index value 1*/
            B, /**< asymetry of the phase function : index value 2*/
            C, /**< fraction of the backward scattering : index value 3*/
            B0,/**< amplitude of the opposition effect : index value 4*/
            H /**< angular width of the opposition effect : index value 5*/
        };

        /**
         * @enum geometry data index enumeration
         */
        enum geometry {
            THETA = 0, /**< view zenith angle : index value 0*/
            THETA_0 = 1, /**< solar zenith angle : index value 1*/
            PSI = 2, /**< azimuth : index value 2*/
            ALPHA = 3, /**< alpha : index value 3*/
            G = 4, /**< phase angle : index value 4*/
            COS_G = 5 /**< cosinus of the phase angle : index value 5*/
        };
    }

/**
 * @namespace ShkutarovEnumeration
 * Enumerations used in the Shkutarov model
 */
    namespace ShkutarovEnumeration {

    }
}
