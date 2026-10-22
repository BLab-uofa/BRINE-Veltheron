import sys
import os
import numpy as np
import random
import copy
import csv
import torch
from botorch.fit import fit_gpytorch_mll
import gpytorch
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel


# Define ExactGPModel as given in the GAUCHE GitHub
class ExactGPModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def transform_inputs(self, X=None):
        return X


class GAUCHE():
    """Class of GAUCHE
    This class can select the next candidates by Bayesian optimization based on GAUCHE package.
    """
    def __init__(self, input_file, output_file, num_objectives, num_proposals):
        """Constructor
        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
        """
        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        
    def load_data(self):
        """Loading candidates
        Returns:
            t_train (list[float]): the list where observed objectives are stored
            X_all (list[float]): the list where all descriptors are stored
            train_actions (list[float]): the list where observed actions are stored
            test_actions (list[float]): the list where test actions are stored
        """
        arr = np.genfromtxt(self.input_file, skip_header=1, delimiter=',')
        arr_train = arr[~np.isnan(arr[:, -1]), :]
        arr_test = arr[np.isnan(arr[:, -1]), :]
        X_train = arr_train[:, :-self.num_objectives]
        t_train = arr_train[:, -self.num_objectives:]
        X_test = arr_test[:, :-self.num_objectives]
        test_actions = np.where(np.isnan(arr[:, -1]))[0].tolist()
        X_all = arr[:, :-self.num_objectives]
        all_actions = [i for i in range(len(X_all))]
        train_actions = np.sort(list(set(all_actions) - set(test_actions)))
        return t_train, X_all, train_actions, test_actions
    
    
    def calc_ai(self, t_train, X_all, train_actions, test_actions):
        """Calculating the proposals by AI algorithm
        This function is for GAUCHE.
        If the new AI algorithm is developed, this function is only changed.
        Args:
            t_train (list[float]): the list where observed objectives are stored
            X_all (list[float]): the list where all descriptors are stored
            train_actions (list[float]): the list where observed actions are stored
            test_actions (list[float]): the list where test actions are stored
        Returns:
            actions (list[int]): the list where the selected actions are stored
        """
        # Define GP likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        train_x = torch.tensor(X_all[train_actions], dtype=torch.float)
        train_y = torch.tensor(t_train.flatten(), dtype=torch.float)
        model = ExactGPModel(train_x, train_y, likelihood)
        # Fit the model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_mll(mll)
        # Set model to evaluation mode
        model.eval()
        likelihood.eval()
        # Select next action using GP predictions (in practice, you may use an acquisition function)
        test_x = torch.tensor(X_all[test_actions], dtype=torch.float)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = model(test_x)
            pred_means = preds.mean
        # Select actions with the highest predicted means (for simplicity)
        selected_indices = torch.topk(pred_means, self.num_proposals).indices.tolist()
        actions = [test_actions[i] for i in selected_indices]
        return actions
    
    
    def select(self):
        """Main function to select the proposals by AI algorithm
        Returns:
            True (str) for success.
        """
        print("Start selection of proposals by GAUCHE!")
        t_train, X_all, train_actions, test_actions = self.load_data()
        actions = self.calc_ai(t_train=t_train, X_all=X_all,
                               train_actions=train_actions, test_actions=test_actions)
        print('Proposals')
        proposals_all = []
        input_data = open(self.input_file, 'r')
        indexes = input_data.readlines()[0].rstrip('\n').split(',')
        indexes = ["actions"] + indexes[0: -self.num_objectives]
        proposals_all.append(indexes)
        
        for i in range(len(actions)):
            row = [str(X_all[actions[i]][j]) for j in range(len(X_all[actions[i]]))]
            row = [str(actions[i])] + row
            proposals_all.append(row)
            print("###")
            print("number =", i + 1)
            print("actions = ", actions[i])
            print("proposal = ", X_all[actions[i]])
            print("###")
        with open(self.output_file, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(proposals_all)
        print("Finish selection of proposals!")
        return "True"