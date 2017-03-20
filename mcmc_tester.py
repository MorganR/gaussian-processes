import numpy as np
import GPflow
from model_tester import ModelTester

class McmcTester(ModelTester):
    def __init__(self, data, num_inducing):
        self.num_classes = np.unique(data.y).size
        self.num_dimensions = data.x.shape[1]

        self.inducing_offset = int(round(data.y.size / num_inducing))
        self.inducing_inputs = data.x[::self.inducing_offset].copy()

        print('Creating SGPMC model with {} classes, {} dimensions, {} images and {} inducing inputs'
            .format(self.num_classes, self.num_dimensions,
            data.y.size, self.inducing_inputs.shape[0]))
        model = GPflow.sgpmc.SGPMC(
            data.x,
            data.y,
            kern=GPflow.kernels.RBF(input_dim=self.num_dimensions),
            likelihood=GPflow.likelihoods.MultiClass(self.num_classes),
            Z=self.inducing_inputs,
            num_latent=self.num_classes)
        model.Z.fixed = True
        super().__init__(data, model)
       
    def train(self):
        print('Training against {} classes using {} images.'.format(self.num_classes, self.data.y.size))
        self.optimize()
        self.export('mcmc-{}c-{}d-{}i-{}z'.format(
            self.num_classes, self.num_dimensions, self.data.y.size, self.inducing_inputs.shape[0]
        ))
