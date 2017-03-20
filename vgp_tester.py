import numpy as np
import GPflow
from model_tester import ModelTester

class VgpTester(ModelTester):
    def __init__(self, data):
        self.num_classes = np.unique(data.y).size
        self.num_dimensions = data.x.shape[1]
        model = GPflow.vgp.VGP(
            data.x,
            data.y,
            kern=GPflow.kernels.RBF(input_dim=self.num_dimensions),
            likelihood=GPflow.likelihoods.MultiClass(self.num_classes),
            num_latent=self.num_classes)
        super().__init__(data, model)
       
    def train(self):
        print('Training against {} classes using {} images.'.format(self.num_classes, self.data.y.size))
        self.optimize()
        self.export('vgp-{}c-{}d-{}i-nopca'.format(
            self.num_classes, self.num_dimensions, self.data.y.size
        ))
