import numpy as np
from galaxies import Galaxies
from galaxy_meta import GalaxyMeta

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
                print('For real: {} guess: {}:'.format(
                    real, guess
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
                
        
        # right_ratios = almost_correct/all_guesses
        # wrong_ratios = almost_wrong/all_guesses
        # accuracy_ratios = (num_test - wrong_guesses)/num_test
        # total_accuracy = (num_test-wrong_guesses.sum())/num_test
        # total_positive_error = np.sqrt(np.sum((all_guesses*right_ratios/num_test)**2))
        # total_negative_error = np.sqrt(np.sum((all_guesses*wrong_ratios/num_test)**2))

        # print('Tested against {} digits with {:.2f}% + {:.2f}% - {:.2f}% accuracy'.format(
        #     num_test, 100*(num_test-wrong_guesses.sum())/num_test, 100*total_positive_error, 100*total_negative_error))
        # for i in range(0,num_test_classes):
        #     print('\tTested {:d} {}s with {:.2f}% + {:.2f}% - {:.2f}% accuracy'.format(
        #         all_guesses[i], 
        #         unique_y[i],
        #         100*(all_guesses[i] - wrong_guesses[i])/all_guesses[i],
        #         100*right_ratios[i],
        #         100*wrong_ratios[i]))

    # def _update_accuracy(i, y_true):
    #     Y_idx = np.argwhere(self.unique_y==y_true)
    #     Y_idx = Y_idx[0]
    #     self.all_guesses[Y_idx] += 1
    #     true_prob = self.p[i, Y_idx]
    #     true_var = self.var[i, Y_idx]
    #     if (self.Y_guess[i] != y_true):
    #         self.wrong_guesses[Y_idx] += 1
    #         highest_prob = self.p[i, self.Y_guess[i]]
    #         if highest_prob < (true_prob + true_var):
    #             self.almost_correct[Y_idx] += 1
    #     else: # guess is correct
    #         if (np.any(self.p[i,np.arange(num_test_classes)!=Y_idx] > (true_prob - true_var))):
    #             self.almost_wrong[Y_idx] += 1