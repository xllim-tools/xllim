import os.path
import unittest
import numpy as np
import math
import kernelo as ker
import matplotlib.pyplot as plt
import pickle


class TestModel_test(unittest.TestCase):
    @staticmethod
    def compute_reconstruction_error(reconstruction, observation):
        return np.linalg.norm(observation - reconstruction) / np.linalg.norm(observation)

    @staticmethod
    def compute_prediction_error(prediction, x):
        return np.linalg.norm(prediction - x) / np.linalg.norm(x)

    #function which returns the index of minimum value in the list
    @staticmethod
    def get_minvalue(inputlist):
        #get the minimum value in the list
        min_value = min(inputlist)
        #return the index of minimum value
        min_index=[]
        for i in range(0,len(inputlist)):
            if min_value == inputlist[i]:
                min_index.append(i)
        return min_index

    def setUp(self) -> None:
        self.datasize = 50000  # 50000
        self.nb_centers = 2
        # Create physical model (here it will be our TestModel)
        self.physicalModel = ker.TestModelConfig().create()
        # Create StatModel
        self.covariances = np.random.uniform(0, 0.0001, self.physicalModel.get_D_dimension())
        self.statModel = ker.GaussianStatModelConfig("sobol", self.physicalModel, self.covariances, 12345).create()
        # Create GLLIM model, including its initialization and training configuration
        self.learningConfig = ker.EMLearningConfig(200, 1e-5, 1e-12)
        self.initConfig = ker.MultInitConfig(seed=123456789, nb_iter_EM=10, nb_experiences=10, gmmLearningConfig=ker.GMMLearningConfig(15, 10, 1e-12))
        self.gllim = ker.GLLiM(self.physicalModel.get_D_dimension(), self.physicalModel.get_L_dimension(), 50, "Full", "Diag", self.initConfig, self.learningConfig)

        if not os.path.isfile("gllim_test_ploynomial_model.file"):
            # Initialize and train GLLIM model
            # Generate synthetic data
            print("Generating dataset")
            self.x_gen, self.y_gen = self.statModel.gen_data(self.datasize)
            print("initializing GLLIM model")
            self.gllim.initialize(self.x_gen, self.y_gen)
            print("done, training model")
            self.gllim.train(self.x_gen, self.y_gen)
            print("traing done")
            self.gllim_parameters = self.gllim.exportModel()
            with open("gllim_test_ploynomial_model.file", "wb") as f:
                pickle.dump(self.gllim_parameters, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            print("GLLIM model saved")
        else:
            with open("gllim_test_ploynomial_model.file", "rb") as f:
                self.gllim_parameters = pickle.load(f)
                self.gllim.importModel(self.gllim_parameters)
            print("GLLIM model loaded")

        # Create predicator
        self.predicator = ker.PredictionConfig(self.nb_centers, self.nb_centers, 1e-10, self.gllim).create()

        # create x_test, y_test dataset
        self.n_samples = 500
        self.x_test_primary_solution = np.zeros((self.n_samples, self.physicalModel.get_L_dimension()))
        self.x_test_secondary_solution = np.zeros((self.n_samples, self.physicalModel.get_L_dimension()))
        self.y_test = np.zeros((self.n_samples, self.physicalModel.get_D_dimension()))
        self.y_test_noised = np.zeros((self.n_samples, self.physicalModel.get_D_dimension()))
        self.y_test_noise = np.zeros((self.n_samples, self.physicalModel.get_D_dimension()))
        self.predictions = []
        self.centerIsPred = []

        for i in range(self.n_samples):
            for j in range(self.physicalModel.get_L_dimension()):
                self.x_test_primary_solution[i, j] = 0.4 * math.sin(2.*math.pi*i/self.n_samples + (j * math.pi/4.)) + 0.5
                if j == 2:
                    self.x_test_secondary_solution[i, j] = 0.4 * math.sin(2.*math.pi*i/self.n_samples + (j * math.pi/4.) + math.pi) + 0.5
                else:
                    self.x_test_secondary_solution[i, j] = 0.4 * math.sin(2.*math.pi*i/self.n_samples + (j * math.pi/4.)) + 0.5

        # fig3, axs3 = plt.subplots(2, 1, constrained_layout=True)
        # axs3[0].plot(self.x_test[:, 2])
        # axs3[1].plot(self.x3_test_secondary_solution[:, 2])
        # plt.show()

        for i in range(self.n_samples):
            self.y_test[i] = self.physicalModel.F(self.x_test_primary_solution[i])
            # Add noise for each Y component
            for j in range(self.physicalModel.get_D_dimension()):
                self.y_test_noise[i][j] = (self.y_test[i][j]/1000.) * np.random.normal(0, math.pow(self.y_test[i][j]/1000., 2), 1)

        self.y_test_noised = self.y_test + self.y_test_noise


    def test_prediction(self):
        # Compute reconstruction mse on centers
        reconstruction_errors = {
            "center1": [],
            "center2": [],
            "min": []
        }
        reconstruction_errors_IS = {
            "center1": [],
            "center2": [],
            "min": []
        }
        y_recontructed_centers = []
        y_reconstructed_centers_IS = []

        # compute predictions
        print("computing predictions")
        for i in range(self.n_samples):
            self.predictions.append(self.predicator.predict(y_obs=self.y_test_noised[i], var_obs=self.y_test_noise[i]))
        self.centerPred = np.array([[pred.centersPred.means[:, k] for pred in self.predictions] for k in range(self.nb_centers)])
        self.predictions = np.array(self.predictions)

        # compute IS
        print("computing IS")
        for nb_center in range(self.nb_centers):
            temp_res_is = []
            for i in range(self.n_samples):
                proposition = ker.GaussianRegularizedPropositionConfig(self.predictions[i].centersPred.means[:, nb_center], self.predictions[i].centersPred.covs[nb_center, :, :]).create()
                sampler = ker.ImportanceSamplingConfig(2000, self.statModel).create()
                temp_res_is.append(sampler.execute(proposition, self.y_test_noised[i], self.y_test_noise[i]))
            self.centerIsPred.append([res_is.mean for res_is in temp_res_is])
        self.centerIsPred = np.array(self.centerIsPred)

        # Compute reconstructed centers
        for i in range(self.nb_centers):
            y_recontructed_centers.append([self.physicalModel.F(x) for x in self.centerPred[i]])
            y_reconstructed_centers_IS.append([self.physicalModel.F(x) for x in self.centerIsPred[i]])
        y_recontructed_centers = np.array(y_recontructed_centers)
        y_reconstructed_centers_IS = np.array(y_reconstructed_centers_IS)

        # Compute errors
        for i in range(self.n_samples):
            for center_nb in range(self.nb_centers):
                reconstruction_errors["center" + str(center_nb+1)].append(self.compute_reconstruction_error(y_recontructed_centers[center_nb, i], self.y_test_noised[i]))
                reconstruction_errors_IS["center" + str(center_nb+1)].append(self.compute_reconstruction_error(y_reconstructed_centers_IS[center_nb, i], self.y_test_noised[i]))
            reconstruction_errors_IS["min"].append(np.amin(np.array([reconstruction_errors_IS["center1"], reconstruction_errors_IS["center2"]])))
            reconstruction_errors["min"].append(np.amin(np.array([reconstruction_errors["center1"], reconstruction_errors["center2"]])))


        # Compute prediction errors
        prediction_errors = {
            "center1": [],
            "center2": []
        }
        prediction_errors_IS = {
            "center1": [],
            "center2": []
        }

        for i in range(self.n_samples):
            current_xobs = None
            for center_nb in range(self.nb_centers):
                error = 0
                min_index = -1
                perm_erros = []
                error_xobs1 = self.compute_prediction_error(self.centerPred[center_nb, i], self.x_test_primary_solution[i])
                error_xobs2 = self.compute_prediction_error(self.centerPred[center_nb, i], self.x_test_secondary_solution[i])
                perm_erros = [error_xobs1, error_xobs2]
                if current_xobs is None:
                    min_index = self.get_minvalue(perm_erros)
                    if not len(min_index) == 1:
                        raise RuntimeError('errors in center predictions permutations have more than one minimim')
                    else:
                        min_index = min_index[0]
                        error = perm_erros[min_index]
                        current_xobs = min_index
                else:
                    error = perm_erros[(current_xobs + 1)%2]
                # print(str(i) + "\t" + "xobs1:\t" + str(error_xobs1) + "\txobs2:\t" + str(error_xobs2) + "\tselected:\t" + str(error))
                prediction_errors["center" + str(center_nb+1)].append(error)

        # Compute prediction IS errors
        for i in range(self.n_samples):
            current_xobs = None
            for center_nb in range(self.nb_centers):
                error = 0
                min_index = -1
                perm_erros = []
                error_xobs1 = self.compute_prediction_error(self.centerIsPred[center_nb, i], self.x_test_primary_solution[i])
                error_xobs2 = self.compute_prediction_error(self.centerIsPred[center_nb, i], self.x_test_secondary_solution[i])
                perm_erros = [error_xobs1, error_xobs2]
                if current_xobs is None:
                    min_index = self.get_minvalue(perm_erros)
                    if not len(min_index) == 1:
                        raise RuntimeError('errors in center predictions permutations have more than one minimim')
                    else:
                        min_index = min_index[0]
                        error = perm_erros[min_index]
                        current_xobs = min_index
                else:
                    error = perm_erros[(current_xobs + 1)%2]
                # print(str(i) + "\t" + "xobs1:\t" + str(error_xobs1) + "\txobs2:\t" + str(error_xobs2) + "\tselected:\t" + str(error))
                prediction_errors_IS["center" + str(center_nb+1)].append(error)


        # Plot
        fig1, axs1 = plt.subplots(2, 2, constrained_layout=True)
        fig1.suptitle('predictions by centers')
        axs1[0, 0].plot(self.x_test_primary_solution[:, 0], 'c,')
        axs1[0, 0].plot(self.centerPred[0, :, 0], 'b.', label='x_center1')
        axs1[0, 0].plot(self.centerPred[1, :, 0], 'g.', label='x_center2')
        axs1[0, 0].set_title('x1')
        axs1[0, 0].set_xlabel('n')
        axs1[0, 0].set_ylabel('x1')
        axs1[0, 0].legend()

        axs1[0, 1].plot(self.x_test_primary_solution[:, 1], 'c,')
        axs1[0, 1].plot(self.centerPred[0, :, 1], 'b.', label='x_center1')
        axs1[0, 1].plot(self.centerPred[1, :, 1], 'g.', label='x_center2')
        axs1[0, 1].set_title('x2')
        axs1[0, 1].set_xlabel('n')
        axs1[0, 1].set_ylabel('x2')
        axs1[0, 1].legend()

        axs1[1, 0].plot(self.x_test_primary_solution[:, 2], 'c,')
        axs1[1, 0].plot(self.centerPred[0, :, 2], 'b.', label='x_center1')
        axs1[1, 0].plot(self.centerPred[1, :, 2], 'g.', label='x_center2')
        axs1[1, 0].plot(self.x_test_secondary_solution[:, 2], 'c,')
        axs1[1, 0].set_title('x3')
        axs1[1, 0].set_xlabel('n')
        axs1[1, 0].set_ylabel('x3')
        axs1[1, 0].legend()

        axs1[1, 1].plot(self.x_test_primary_solution[:, 3], 'c,')
        axs1[1, 1].plot(self.centerPred[0, :, 3], 'b.', label='x_center1')
        axs1[1, 1].plot(self.centerPred[1, :, 3], 'g.', label='x_center2')
        axs1[1, 1].set_title('x4')
        axs1[1, 1].set_xlabel('n')
        axs1[1, 1].set_ylabel('x4')
        axs1[1, 1].legend()

        fig2, axs2 = plt.subplots(2, 2, constrained_layout=True)
        fig2.suptitle('predictions by centers IS')
        axs2[0, 0].plot(self.x_test_primary_solution[:, 0], 'c,')
        axs2[0, 0].plot(self.centerIsPred[0, :, 0], 'r.', label='x_is 1')
        axs2[0, 0].plot(self.centerIsPred[1, :, 0], 'm.', label='x_is 2')
        axs2[0, 0].set_title('x1')
        axs2[0, 0].set_xlabel('n')
        axs2[0, 0].set_ylabel('x1')
        axs2[0, 0].legend()

        axs2[0, 1].plot(self.x_test_primary_solution[:, 1], 'c,')
        axs2[0, 1].plot(self.centerIsPred[0, :, 1], 'r.', label='x_is 1')
        axs2[0, 1].plot(self.centerIsPred[1, :, 1], 'm.', label='x_is 2')
        axs2[0, 1].set_title('x2')
        axs2[0, 1].set_xlabel('n')
        axs2[0, 1].set_ylabel('x2')
        axs2[0, 1].legend()

        axs2[1, 0].plot(self.x_test_primary_solution[:, 2], 'c,')
        axs2[1, 0].plot(self.x_test_secondary_solution[:, 2], 'c,')
        axs2[1, 0].plot(self.centerIsPred[0, :, 2], 'r.', label='x_is 1')
        axs2[1, 0].plot(self.centerIsPred[1, :, 2], 'm.', label='x_is 2')
        axs2[1, 0].set_title('x3')
        axs2[1, 0].set_xlabel('n')
        axs2[1, 0].set_ylabel('x3')
        axs2[1, 0].legend()

        axs2[1, 1].plot(self.x_test_primary_solution[:, 3], 'c,')
        axs2[1, 1].plot(self.centerIsPred[0, :, 3], 'r.', label='x_is 1')
        axs2[1, 1].plot(self.centerIsPred[1, :, 3], 'm.', label='x_is 2')
        axs2[1, 1].set_title('x4')
        axs2[1, 1].set_xlabel('n')
        axs2[1, 1].set_ylabel('x4')
        axs2[1, 1].legend()

        fig3, axs3 = plt.subplots(1, 1, constrained_layout=True)
        fig3.suptitle('Reconstruction error')
        axs3.set_yscale('log')
        axs3.plot(reconstruction_errors["center1"], 'b.', label='center1')
        axs3.plot(reconstruction_errors["center2"], 'g.', label='center2')
        axs3.plot(reconstruction_errors_IS["center1"], 'r.', label='center1_is')
        axs3.plot(reconstruction_errors_IS["center2"], 'm.', label='center2_is')
        axs3.legend()
        plt.grid()

        fig4, axs4 = plt.subplots(1, 1, constrained_layout=True)
        fig4.suptitle('Prediction error')
        axs4.set_yscale('log')
        axs4.plot(prediction_errors["center1"], 'b.', label='center1')
        axs4.plot(prediction_errors["center2"], 'g.', label='center2')
        axs4.plot(prediction_errors_IS["center1"], 'r.', label='center1_is')
        axs4.plot(prediction_errors_IS["center2"], 'm.', label='center2_is')
        axs4.legend()
        plt.grid()
        plt.show()


    def test_is(self):
        b = 0

if __name__ == '__main__':
    unittest.main()
