import numpy as np

class ShkuratovModel5p(object):
    """ This is a python class defining the Shkuratov physical model. 
    
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

    DEGREE_180 = 180
    L_dimension = 5
    scalingCoeffs = [1.0,1.5,1.5,1.5,1.5]
    offset = [0,0,0.2,0,0]

    # geometry
    ALPHA = 0
    BETA = 1
    GAMMA = 2
    INC = 0 # sza
    EME = 1 # vza
    PHI = 2 # phi

    # photometry
    AN = 0
    MU_1 = 1
    NU = 2
    M = 3
    MU_2 = 4
 
    #################################################################################################
    ##                          CORE FUNCTIONS (MANDATORY)                                         ##
    #################################################################################################

    def F(self, photometry):
        photometry = self.toPhysic(photometry)

        f = (np.exp(- photometry[self.MU_1] * self.configuredGeometries[self.ALPHA]) + photometry[self.M] * np.exp(- photometry[self.MU_2] * self.configuredGeometries[self.ALPHA])) / (1 + photometry[self.M])
        d = np.cos(self.configuredGeometries[self.ALPHA] / 2.0) * np.cos(np.pi * (self.configuredGeometries[self.GAMMA] - self.configuredGeometries[self.ALPHA] / 2.0) / (np.pi - self.configuredGeometries[self.ALPHA])) / np.cos(self.configuredGeometries[self.GAMMA])
        o=d*np.cos(self.configuredGeometries[self.BETA])**(photometry[self.NU] * self.configuredGeometries[self.ALPHA] * (np.pi - self.configuredGeometries[self.ALPHA]))
        reflectances = photometry[self.AN] * o.T * f.T / self.cos_i.T

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
        # 71 geometries
        geom_tmp.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,40,40,40,40,40,40,40,40,40,40,40,40,40,40,60,60,60,60,60,60,60,60,60,60,60,60,60,60,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20])
        geom_tmp.append([70,60,50,40,30,20,10,10,20,30,40,50,60,70,70,60,50,40,30,20,10,0,10,30,40,50,60,70,70,60,50,30,20,10,0,10,20,30,40,50,60,70,70,60,50,40,30,20,10,0,10,20,30,40,50,70,70,60,50,40,30,20,10,0,10,20,30,40,50,60,70])
        geom_tmp.append([0,0,0,0,0,0,0,180,180,180,180,180,180,180,180,180,180,180,180,180,180,0,0,0,0,0,0,0,0,0,0,0,0,0,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,0,0,0,0,0,0,30,30,30,30,30,30,30,150,150,150,150,150,150,150,150])
        # 70 geometries
        inc = [
            26.543688820608587,
            59.70151372254296,
            49.75026883595537,
            22.25687341017779,
            29.37477052338286,
            45.92667930975748,
            0.8378149317275863,
            0.9539443924366908,
            50.6751036681658,
            30.939391754687584,
            45.35750723379789,
            77.4098063051496,
            8.550137962131798,
            78.75041993019249,
            57.21323807619601,
            64.45682060312284,
            61.27808641039031,
            29.87156688024389,
            58.23236626065648,
            0.46252284502360075,
            61.414629361746286,
            68.38870017628031,
            26.738228752953002,
            62.23719667072418,
            37.97518925487721,
            11.71217249013775,
            44.21083972798597,
            54.80504984567808,
            6.681918781680803,
            77.75380284700469,
            18.248735054007334,
            80.74809445945607,
            27.883083585044094,
            30.667505907879136,
            23.753533303937083,
            79.98537185251999,
            57.116427002329445,
            11.093398071099992,
            21.05442110717268,
            39.39417817217474,
            67.3088045536496,
            45.67330930572437,
            66.22342831254535,
            16.299202757933486,
            23.17100000877112,
            33.81214522345879,
            5.674512543132457,
            46.31197332927556,
            4.425076559963095,
            43.39587018527353
        ]
        eme = [
            57.82353475394024,
            75.1388961438802,
            53.34082125952007,
            59.16388136864589,
            56.82524519395523,
            13.861455932687909,
            61.01513462943814,
            0.08022416723222348,
            45.20326439963487,
            49.75804377781624,
            47.57479946505285,
            40.83414539871288,
            42.582107181671304,
            19.543259862080465,
            79.42231822683355,
            5.608973502992881,
            8.519518228991334,
            14.653496103650852,
            83.62967972867986,
            63.630581768018686,
            50.97355037385619,
            43.39685725973246,
            75.09920079639205,
            64.31108516343195,
            25.878527299490525,
            59.95252508789103,
            7.817590072337685,
            63.917416324181204,
            49.224108815946195,
            11.8651829920254,
            52.905039469008344,
            40.1809318771301,
            18.01508428188126,
            26.55789502901616,
            58.87354290360373,
            27.541622847565314,
            27.640845197335114,
            7.317915528927937,
            27.26977361165818,
            43.424177588459685,
            50.72108586970755,
            21.515785742893073,
            58.810741521507,
            74.65446772899669,
            65.1588001626688,
            57.74835037606868,
            40.286095753565505,
            80.72344684958227,
            46.045282404078506,
            31.16502580002508
        ]
        phi =[
            56.860734390493455,
            73.9693641883178,
            37.73213502869909,
            115.17115948730651,
            173.9808170538202,
            27.381659255810384,
            145.5789039136927,
            161.34338905048256,
            69.03017040464387,
            42.873394998955916,
            51.61878013029947,
            18.286412087024594,
            94.0626977818526,
            14.107568014786057,
            63.49011301889791,
            51.18375024794363,
            133.34658272658174,
            157.10530541937266,
            146.0687864919587,
            166.71021816930005,
            175.97262739732273,
            77.64637076284835,
            113.89679858896427,
            90.10458760238879,
            178.93891790686803,
            139.34645381672362,
            103.98331202999472,
            153.69553818327597,
            35.75205008184436,
            172.68559130787182,
            112.44694565309588,
            66.94556052034761,
            142.26012240479503,
            135.8978116867964,
            144.40908529405644,
            21.29090671425231,
            103.1093776827147,
            44.744140970189484,
            175.96850407152732,
            59.46378165314184,
            71.48685119783887,
            144.2711967755729,
            113.06470564914261,
            26.905165233458316,
            123.66223897723546,
            5.7269090111509335,
            157.16123545573817,
            76.01843764637619,
            112.82685894796201,
            117.10375378471507
        ]
        geom_tmp = [inc,eme,phi]  
        
        geometries=np.array(geom_tmp)
        self.D_dimension=geometries.shape[1]
        self.configuredGeometries=self.setupGeometries(geometries)
        self.cos_i=np.cos(geometries[self.INC]*np.pi / self.DEGREE_180)

    def degToGrad(self, degree):
        return degree * np.pi / self.DEGREE_180


    def setupGeometries(self, geometries):
        configuredGeometries = np.zeros(geometries.shape)
        geomsGrad = np.array(geometries)
        geomsGrad = self.degToGrad(geomsGrad)

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
        #tmp_angle=np.arccos(np.cos(geomsGrad[self.EME]) / cos_beta)
        #configuredGeometries[self.GAMMA] = np.where(geomsGrad[self.PHI]<ma.pi/2.,-tmp_angle,tmp_angle)
        configuredGeometries[self.GAMMA] = np.arctan((np.cos(geomsGrad[self.INC])/np.cos(geomsGrad[self.EME])-np.cos(configuredGeometries[self.ALPHA]))/np.sin(configuredGeometries[self.ALPHA]))

        return configuredGeometries
