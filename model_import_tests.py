import numpy as np
import matplotlib.pyplot as plt
from model_tester import ModelTester
from data_holder import DataHolder

models = [  'vgp-10c-2d-250i',
            'vgp-10c-3d-250i',
            'vgp-10c-5d-250i',
            'vgp-10c-10d-250i',
            'vgp-10c-15d-250i',
            'vgp-10c-25d-250i',
            'vgp-10c-50d-250i',
            'vgp-10c-100d-250i',
            'vgp-10c-150d-250i',
            'vgp-10c-250d-250i'
        ]

for m_str in models:
    m_test = ModelTester(DataHolder([0], [1], [0], [1]), None)
    m_test.import_old(m_str)
    m_test.test(10000)