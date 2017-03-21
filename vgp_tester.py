import numpy as np
import GPflow
from model_tester import ModelTester

class VgpTester(ModelTester):
    def __init__(self, data, kern_id):
        self.num_classes = np.unique(data.y).size
        self.num_dimensions = data.x.shape[1]
        self.kern_id = kern_id

        if kern_id == 'rbf':
            kernel = GPflow.kernels.RBF(input_dim=self.num_dimensions, ARD=True, lengthscales=0.1*np.ones(self.num_dimensions))
        elif kern_id == 'm12':
            kernel = GPflow.kernels.Matern12(input_dim=self.num_dimensions, ARD=True, lengthscales=0.1*np.ones(self.num_dimensions))
        elif kern_id == 'm32':
            kernel = GPflow.kernels.Matern32(input_dim=self.num_dimensions, ARD=True, lengthscales=0.1*np.ones(self.num_dimensions))
        elif kern_id == 'm52':
            kernel = GPflow.kernels.Matern52(input_dim=self.num_dimensions, ARD=True, lengthscales=0.1*np.ones(self.num_dimensions))

        model = GPflow.vgp.VGP(
            data.x,
            data.y,
            kern=kernel,
            likelihood=GPflow.likelihoods.MultiClass(self.num_classes),
            num_latent=self.num_classes)
        super().__init__(data, model)
       
    def train(self):
        print('Training against {} classes using {} images.'.format(self.num_classes, self.data.y.size))
        self.optimize()
        self.export('vgp-{}c-{}d-{}i-'.format(
            self.num_classes, self.num_dimensions, self.data.y.size
        ) + self.kern_id)
