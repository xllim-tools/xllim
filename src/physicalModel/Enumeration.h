/**
 * @file Enumeration.h
 * @brief This file contains different enumeration used in the functional model
 * @author Sami DJOUADI
 * @version 1.0
 * @date 18/12/2019
 */

/**
 * @namespace HapkeEnumeration
 * Enumerations used in the Hapke model
 */
namespace HapkeEnumeration{

    /**
     * @enum photometry data index enumeration
     */
    enum photometry{
        OMEGA, /**< enum value 0*/
        THETA_BAR, /**< enum value 1*/
        B, /**< enum value 2*/
        C,/**< enum value 3*/
        B0,/**< enum value 4*/
        H /**< enum value 5*/
    };

    /**
     * @enum geometry data index enumeration
     */
    enum geometry{
        THETA = 0, /**< enum value 0*/
        THETA_0 = 1, /**< enum value 1*/
        PSI = 2, /**< enum value 2*/
        ALPHA = 3, /**< enum value 3*/
        G = 4, /**< enum value 4*/
        COS_G = 5 /**< enum value 5*/
    };
}

namespace ShkutarovEnumeration{

}

