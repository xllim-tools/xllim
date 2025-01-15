import numpy as np
import json
import os

class RPVModel(object):
    """ This is a python class defining the RPV physical model. 
    
    This class is composed of 5 mandatory functions:
        - F: the functional model F describing the physical model. F takes photometries as arguments and return reflectances
        - getDimensionY: returns the dimension of Y (reflectances)
        - getDimensionX: return de dimension of X (photometries)
        - toPhysic: converts the X data from mathematical framework (0<X<1) to physical framework
        - fromPhysic: converts the X data from physical framework to mathematical framework (0<X<1)

    Note that some class constants, other functions and class constructors can be declared.
    """
    
    #################################################################################################
    ##                          CLASS CONSTANTS (OPTIONAL)                                         ##
    #################################################################################################

    L_dimension = 3
    scalingCoeffs = [1.0,1.2,2.0]
    offset = [0.,0.,-1.]

    # geometry
    ALPHA = 0
    BETA = 1
    GAMMA = 2
    INC = 0 # sza
    EME = 1 # vza
    PHI = 2 # phi

    # photometry
    rho_0 = 0
    K = 1
    G = 2
 
    #################################################################################################
    ##                          CORE FUNCTIONS (MANDATORY)                                         ##
    #################################################################################################

    def F(self, photometry):
        photometry = self.toPhysic(photometry)

        cos_i = np.cos(np.radians(self.geometries[self.INC]))
        cos_e = np.cos(np.radians(self.geometries[self.EME]))
        tan_i= np.tan(np.radians(self.geometries[self.INC]))
        tan_e = np.tan(np.radians(self.geometries[self.EME]))
        cos_phi=np.cos(np.radians(self.geometries[self.PHI]))
    
        f = (1-photometry[self.G]**2)/(1+photometry[self.G]**2+2.*photometry[self.G]*np.cos(self.configuredGeometries[self.ALPHA]))**1.5

        m = (cos_i**(photometry[self.K]-1) * cos_e**(photometry[self.K]-1)) / (cos_i+cos_e)**(1-photometry[self.K])

        o = (tan_i**2 + tan_e**2 -2*tan_i*tan_e*cos_phi)**0.5

        h = 1+ (1-photometry[self.rho_0])/(1+o)

        reflectances = photometry[self.rho_0] * f.T * m.T * h.T

        return reflectances


    def getDimensionY(self):
        return self.D_dimension

    def getDimensionX(self):
        return self.L_dimension

    def toPhysic(self, x):
        for l in range(x.shape[0]):
            x[l] = x[l] * self.scalingCoeffs[l] + self.offset[l]
        return x

    def fromPhysic(self, x):
        for l in range(x.shape[0]):
            x[l] = (x[l] - self.offset[l]) / self.scalingCoeffs[l]
        return x


    #################################################################################################
    ##                          OTHER FUNCTIONS (OPTIONAL)                                         ##
    #################################################################################################

    def __init__(self):
        # geometries data
        geom_tmp = []
        geom_tmp.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,40,40,40,40,40,40,40,40,40,40,40,40,40,40,60,60,60,60,60,60,60,60,60,60,60,60,60,60,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20])
        geom_tmp.append([70,60,50,40,30,20,10,10,20,30,40,50,60,70,70,60,50,40,30,20,10,0,10,30,40,50,60,70,70,60,50,30,20,10,0,10,20,30,40,50,60,70,70,60,50,40,30,20,10,0,10,20,30,40,50,70,70,60,50,40,30,20,10,0,10,20,30,40,50,60,70])
        geom_tmp.append([0,0,0,0,0,0,0,180,180,180,180,180,180,180,180,180,180,180,180,180,180,0,0,0,0,0,0,0,0,0,0,0,0,0,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,0,0,0,0,0,0,30,30,30,30,30,30,30,150,150,150,150,150,150,150,150])
        geometries=np.array(geom_tmp)
        self.D_dimension=geometries.shape[1]
        self.configuredGeometries=self.setupGeometries(geometries)
        self.geometries=geometries

    def setupGeometries(self, geometries):
        configuredGeometries = np.zeros(geometries.shape)
        geomsGrad = np.array(geometries)
        geomsGrad = np.radians(geomsGrad)

        #compute Alpha
        configuredGeometries[self.ALPHA] = np.arccos(np.cos(geomsGrad[self.INC]) * np.cos(geomsGrad[self.EME]) + np.sin(geomsGrad[self.INC]) * np.sin(geomsGrad[self.EME]) * np.cos(geomsGrad[self.PHI]))

        #compute Beta
        sin_i_e_2 = pow(np.sin(geomsGrad[self.INC] + geomsGrad[self.EME]),2)
        cos_phiDiv2_2 = pow(np.cos(geomsGrad[self.PHI]/2.0),2)
        sin_2_i = np.sin(geomsGrad[self.INC] * 2)
        sin_2_e = np.sin(geomsGrad[self.EME] * 2)
        cos_beta = np.sqrt(
                (sin_i_e_2 - cos_phiDiv2_2 * sin_2_i * sin_2_e) /
                (sin_i_e_2 - cos_phiDiv2_2 * sin_2_i * sin_2_e + pow(np.sin(geomsGrad[self.EME]),2) * pow(np.sin(geomsGrad[self.INC]),2) * pow(np.sin(geomsGrad[self.PHI]),2)))
        configuredGeometries[self.BETA] = np.arccos(cos_beta)

        #compute Gamma
        configuredGeometries[self.GAMMA] = np.arccos(np.cos(geomsGrad[self.EME]) / cos_beta)

        return configuredGeometries
