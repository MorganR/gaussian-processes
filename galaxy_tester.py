import numpy as np
from galaxies import Galaxies
from galaxy_meta import GalaxyMeta
import matplotlib.pyplot as plt

class GalaxyTester():
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.galaxies = Galaxies()
        self.test_ids = self.galaxies.get_flattened_test_ids()
        self.meta = GalaxyMeta()
        
    def test(self):
        num_test = self.data.y_test.size
        num_test_classes = np.unique(self.data.y_test).size
        num_train_classes = np.unique(self.data.y).size

        self.p, self.var = self.model.predict_y(self.data.x_test[:num_test])

        self.Y_guess = np.argmax(self.p, axis=1)
        # self.all_guesses = np.zeros(num_test_classes, dtype=np.int32)
        # self.wrong_guesses = np.zeros(num_test_classes, dtype=np.int32)
        # self.almost_correct = np.zeros(num_test_classes, dtype=np.int32)
        # self.almost_wrong = np.zeros(num_test_classes, dtype=np.int32)
        # self.unique_y = np.sort(np.unique(self.data.y))

        self.idx = [
            [[],[]],
            [[],[]],
            [[],[]]
            ]
        self.prob = [
            [[],[]],
            [[],[]],
            [[],[]]
        ]
        self.variances = [
            [[],[]],
            [[],[]],
            [[],[]]
        ]
        self.objid = [
            [[],[]],
            [[],[]],
            [[],[]]
            ]
        self.cs = [
            [[],[]],
            [[],[]],
            [[],[]]
            ]
        self.el = [
            [[],[]],
            [[],[]],
            [[],[]]
            ]
        self.nvote = [
            [[],[]],
            [[],[]],
            [[],[]]
            ]
        self.edge = [
            [[],[]],
            [[],[]],
            [[],[]]
            ] 
        self.disk = [
            [[],[]],
            [[],[]],
            [[],[]]
            ]
        self.merge = [
            [[],[]],
            [[],[]],
            [[],[]]
            ]
        self.acw = [
            [[],[]],
            [[],[]],
            [[],[]]
            ]
        for real in range(0, num_test_classes):
            real_idx = self.data.y_test == real
            for guess in range(0, num_train_classes):
                guess_idx = self.Y_guess == guess
                self.idx[real][guess] = real_idx & guess_idx
                self.prob[real][guess] = self.p[self.idx[real][guess],:]
                self.variances[real][guess] = self.var[self.idx[real][guess],:]
                self.objid[real][guess] = self.test_ids[self.idx[real][guess]]
                for objid in self.objid[real][guess]:
                    spiral, elliptical, unc, n, ra, dec, cs, el, merge, edge, acw, cw, disk = self.meta.find_by_id(objid, real)
                    self.cs[real][guess].append(cs)
                    self.el[real][guess].append(el)
                    self.nvote[real][guess].append(n)
                    self.edge[real][guess].append(edge)
                    self.disk[real][guess].append(disk)
                    self.merge[real][guess].append(merge)
                    self.acw[real][guess].append(acw)
                print('For real: {} guess: {} ({} images):'.format(
                    real, guess, len(self.nvote[real][guess])
                ))
                print('\tprob 0: {} ({})'.format(
                    np.average(self.prob[real][guess][:,0]),
                    np.std(self.prob[real][guess][:,0])
                ))
                print('\tprob 1: {} ({})'.format(
                    np.average(self.prob[real][guess][:,1]),
                    np.std(self.prob[real][guess][:,1])
                ))
                print('\tnvotes: {} ({})'.format(
                    np.average(self.nvote[real][guess]),
                    np.std(self.nvote[real][guess])
                ))
                print('\tcs: {} ({})'.format(
                    np.average(self.cs[real][guess]),
                    np.std(self.cs[real][guess])
                ))
                print('\tel: {} ({})'.format(
                    np.average(self.el[real][guess]),
                    np.std(self.el[real][guess])
                ))
                print('\tspiral: {} ({})'.format(
                    np.average(self.spiral[real][guess]),
                    np.std(self.spiral[real][guess])
                ))
                print('\telliptical: {} ({})'.format(
                    np.average(self.elliptical[real][guess]),
                    np.std(self.elliptical[real][guess])
                ))
                print('\tedge: {} ({})'.format(
                    np.average(self.edge[real][guess]),
                    np.std(self.edge[real][guess])
                ))
                print('\tdisk: {} ({})'.format(
                    np.average(self.disk[real][guess]),
                    np.std(self.disk[real][guess])
                ))
                print('\tmerge: {} ({})'.format(
                    np.average(self.merge[real][guess]),
                    np.std(self.merge[real][guess])
                ))
                print('\tacw: {} ({})'.format(
                    np.average(self.acw[real][guess]),
                    np.std(self.acw[real][guess])
                ))


        labels = {0: 'Spiral', 1:'Elliptical', 2:'Uncertain'}
        
        real = 0
        not_real = 1
        probs = 0
        votes = self.cs

        plt.errorbar(self.prob[real][real][:,probs].flatten()*100,
            np.reshape(votes[real][real], len(votes[real][real]))*100, 
            xerr=self.variances[real][real][:,probs].flatten()*100,
            errorevery=10,
            fmt='.', label=labels[real])
        plt.errorbar(self.prob[real][not_real][:,probs].flatten()*100,
            np.reshape(votes[real][not_real], len(votes[real][not_real]))*100,
            xerr=self.variances[real][not_real][:,probs].flatten()*100,
            errorevery=10,
            fmt='.', label=labels[not_real])
        plt.title('Comparison of GP Model Classification Probability'+
            '\nWith with Galaxy Zoo Votes for ' + labels[real] + ' Galaxies')
        plt.xlabel('GP Model ' + labels[probs] + ' Probability (%)')
        plt.ylabel('Galaxy Zoo ' + labels[real] + ' Votes (%)')
        fig = plt.gcf()
        fig.savefig(labels[real] + '-' + labels[probs]+'.eps')
        plt.legend(loc='lower right')
        plt.show()

        # plt.plot(self.prob[0][0][:,1].flatten()*100, np.reshape(self.el[0][0], len(self.el[0][0]))*100, '.', label='Spiral')
        # plt.plot(self.prob[0][1][:,1].flatten()*100, np.reshape(self.el[0][1], len(self.el[0][1]))*100, '.', label='Elliptical')
        # plt.title('Comparison of GP Model Classification Probability'+
        #     '\nWith with Galaxy Zoo Votes for Elliptical Galaxies')
        # plt.xlabel('GP Model Elliptical Probability (%)')
        # plt.ylabel('Galaxy Zoo Elliptical Votes (%)')
        # fig = plt.gcf()
        # fig.savefig('spiral-elliptical.eps')
        # plt.legend()
        # plt.show()

        # plt.plot(self.prob[1][0][:,0].flatten()*100, np.reshape(self.cs[1][0], len(self.cs[1][0]))*100, '.', label='Spiral')
        # plt.plot(self.prob[1][1][:,0].flatten()*100, np.reshape(self.cs[1][1], len(self.cs[1][1]))*100, '.', label='Elliptical')
        # plt.title('Comparison of GP Model Classification Probability'+
        #     '\nWith with Galaxy Zoo Votes for Spiral Galaxies')
        # plt.xlabel('GP Model Spiral Probability (%)')
        # plt.ylabel('Galaxy Zoo Spiral Votes (%)')
        # fig = plt.gcf()
        # fig.savefig('elliptical-spiral.eps')
        # plt.legend()
        # plt.show()

        real = 1
        not_real = 0
        probs = 1
        votes = self.el

        plt.errorbar(self.prob[real][real][:,probs].flatten()*100,
            np.reshape(votes[real][real], len(votes[real][real]))*100, 
            xerr=self.variances[real][real][:,probs].flatten()*100,
            errorevery=10,
            fmt='.', label=labels[real])
        plt.errorbar(self.prob[real][not_real][:,probs].flatten()*100,
            np.reshape(votes[real][not_real], len(votes[real][not_real]))*100,
            xerr=self.variances[real][not_real][:,probs].flatten()*100,
            errorevery=10,
            fmt='.', label=labels[not_real])
        plt.title('Comparison of GP Model Classification Probability'+
            '\nWith with Galaxy Zoo Votes for ' + labels[real] + ' Galaxies')
        plt.xlabel('GP Model ' + labels[probs] + ' Probability (%)')
        plt.ylabel('Galaxy Zoo ' + labels[real] + ' Votes (%)')
        fig = plt.gcf()
        fig.savefig(labels[real] + '-' + labels[probs]+'.eps')
        plt.legend(loc='lower right')
        plt.show()
        
        real = 2
        not_probs = 0
        probs = 1
        votes = self.el

        plt.errorbar(self.prob[real][probs][:,probs].flatten()*100,
            np.reshape(votes[real][probs], len(votes[real][probs]))*100, 
            xerr=self.variances[real][probs][:,probs].flatten()*100,
            errorevery=10,
            fmt='.', label=labels[real])
        plt.errorbar(self.prob[real][not_probs][:,probs].flatten()*100,
            np.reshape(votes[real][not_probs], len(votes[real][not_probs]))*100,
            xerr=self.variances[real][not_probs][:,probs].flatten()*100,
            errorevery=10,
            fmt='.', label=labels[not_probs])
        plt.title('Comparison of GP Model Classification Probability'+
            '\nWith with Galaxy Zoo Votes for ' + labels[real] + ' Galaxies')
        plt.xlabel('GP Model ' + labels[probs] + ' Probability (%)')
        plt.ylabel('Galaxy Zoo ' + labels[real] + ' Votes (%)')
        fig = plt.gcf()
        fig.savefig(labels[real] + '-' + labels[probs]+'.eps')
        plt.legend(loc='lower right')
        plt.show()

    # def plot_ratios(real, numerator, title):
    #     labels = {0: 'Spiral', 1:'Elliptical', 2:'Uncertain'}
    #     denom = 0
    #     if (numerator == 0):
    #         denom = 1
    #     plt.plot(self.prob[real][real]/self.prob[real][])