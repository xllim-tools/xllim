import unittest
import numpy as np
import math
import kernelo as ker
import matplotlib.pyplot as plt

class TestModel_test(unittest.TestCase):
    def setUp(self) -> None:
        self.datasize = 500000  # 50000
        # Create physical model (here it will be our TestModel)
        self.physicalModel = ker.TestModelConfig().create()

        # Create StatModel
        self.covariances = np.random.uniform(0, 0.0001, self.physicalModel.get_D_dimension())
        self.statModel = ker.GaussianStatModelConfig("sobol", self.physicalModel, self.covariances, 12345).create()

        # Generate synthetic data
        self.x_gen, self.y_gen = self.statModel.gen_data(self.datasize)

        # Create GLLIM model, including its initialization and training configuration
        self.learningConfig = ker.EMLearningConfig(200, 1e-5, 1e-12)
        self.initConfig = ker.MultInitConfig(seed=123456789, nb_iter_EM=10, nb_experiences=10, gmmLearningConfig=ker.GMMLearningConfig(15, 10, 1e-12))
        self.gllim = ker.GLLiM(self.physicalModel.get_D_dimension(), self.physicalModel.get_L_dimension(), 50, "Full", "Diag", self.initConfig, self.learningConfig)

        # Initialize and train GLLIM model
        print("initializing GLLIM model")
        self.gllim.initialize(self.x_gen, self.y_gen)
        print("done, training model")
        self.gllim.train(self.x_gen, self.y_gen)
        print("traing done")
        self.predicator = ker.PredictionConfig(2, 2, 1e-10, self.gllim).create()  # threshold to be determined by sylvain

        # create x_test, y_test dataset
        self.n_samples = 500
        self.x_test = np.zeros((self.n_samples, self.physicalModel.get_L_dimension()))
        self.y_test = np.zeros((self.n_samples, self.physicalModel.get_D_dimension()))
        self.y_test_noised = np.zeros((self.n_samples, self.physicalModel.get_D_dimension()))
        self.y_test_noise = np.zeros((self.n_samples, self.physicalModel.get_D_dimension()))
        self.predictions = []
        self.res_is1 = []
        for i in range(self.n_samples):
            for j in range(self.physicalModel.get_L_dimension()):
                self.x_test[i, j] = 0.4 * math.sin(2.*math.pi*i/self.n_samples + (j * math.pi/4.)) + 0.5

        for i in range(self.n_samples):
            self.y_test[i] = self.physicalModel.F(self.x_test[i])
            # Add noise for each Y component
            for j in range(self.physicalModel.get_D_dimension()):
                self.y_test_noise[i][j] = (self.y_test[i][j]/1000.) * np.random.normal(0, math.pow(self.y_test[i][j]/1000., 2), 1)
        self.y_test_noised = self.y_test + self.y_test_noise

        # compute predictions
        print("computing predictions")
        for i in range(self.n_samples):
            self.predictions.append(self.predicator.predict(y_obs=self.y_test_noised[i], var_obs=self.y_test_noise[i]))
        self.centerMeans = [pred.centersPred.means for pred in self.predictions]

        # compute IS
        print("computing IS")
        for i in range(self.n_samples):
            proposition = ker.GaussianMixturePropositionConfig(self.predictions[i].meansPred.gmm_weights, self.predictions[i].meansPred.gmm_means, self.predictions[i].meansPred.gmm_covs).create()
            sampler = ker.ImportanceSamplingConfig(2000, self.statModel).create()
            self.res_is1.append(sampler.execute(proposition, self.y_test_noised[i], self.y_test_noise[i]))
        self.res_is1_centers = [res_is.mean for res_is in self.res_is1]

        self.predictions = np.array(self.predictions)
        self.centerMeans = np.array(self.centerMeans)
        self.res_is1_centers = np.array(self.res_is1_centers)

    # def test_functional(self):
    #     y_F = [self.physicalModel.F(x) for x in self.x_gen]
    #     delta = 100 * abs(self.y_gen - y_F)
    #     self.assertTrue(np.amax(delta) < 1e-5)

    def test_prediction(self):
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        fig.suptitle('predictions')
        axs[0, 0].plot(self.x_test[:, 0])
        axs[0, 0].plot(self.centerMeans[:, 0, :], '.', label='x_center')
        axs[0, 0].plot(self.res_is1_centers[:, 0], 'r.', label='x_is')
        axs[0, 0].set_title('1')
        axs[0, 0].set_xlabel('n')
        axs[0, 0].set_ylabel('x1')
        axs[0, 0].legend()

        axs[0, 1].plot(self.x_test[:, 1])
        axs[0, 1].plot(self.centerMeans[:, 1, :], '.', label='x_center')
        axs[0, 1].plot(self.res_is1_centers[:, 1], 'r.', label='x_is')
        axs[0, 1].set_title('2')
        axs[0, 1].set_xlabel('n')
        axs[0, 1].set_ylabel('x2')
        axs[0, 1].legend()

        axs[1, 0].plot(self.x_test[:, 2])
        axs[1, 0].plot(self.centerMeans[:, 2, :], '.', label='x_center')
        axs[1, 0].plot(self.res_is1_centers[:, 2], 'r.', label='x_is')
        axs[1, 0].set_title('3')
        axs[1, 0].set_xlabel('n')
        axs[1, 0].set_ylabel('x3')
        axs[1, 0].legend()

        axs[1, 1].plot(self.x_test[:, 3])
        axs[1, 1].plot(self.centerMeans[:, 3, :], '.', label='x_center')
        axs[1, 1].plot(self.res_is1_centers[:, 3], 'r.', label='x_is')
        axs[1, 1].set_title('4')
        axs[1, 1].set_xlabel('n')
        axs[1, 1].set_ylabel('x4')
        axs[1, 1].legend()

        plt.show()

# plt.figure()
        # plt.plot(self.y_test_noised)
        # plt.figure()
        # plt.plot(self.y_test)
        # plt.show()

    def test_is(self):
        b = 0

if __name__ == '__main__':
    unittest.main()
