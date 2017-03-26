import numpy as np

class GalaxyMeta():
    def __init__(self):
        self.base_files = ['spirals', 'ellipticals', 'uncertains', 'spiral-edges', 'uncertain-edges']
        files = ('galaxies/10000-'+bf for bf in self.base_files)
        self.objid = {}
        self.spiral = {}
        self.elliptical = {}
        self.uncertain = {}
        self.nvote = {}
        self.ra = {}
        self.dec = {}
        self.cs_debiased = {}
        self.el_debiased = {}
        self.p_merge = {}
        self.p_edge = {}
        self.p_acw = {}
        self.p_cw = {}
        self.p_disk = {}

        for bf,f in zip(self.base_files, files):
            objid, spiral, elliptical, uncertain, nvote = np.loadtxt(
                f+'.csv', skiprows=2, usecols=[0, 3, 4, 5, 6], delimiter=',', unpack=True, dtype=np.int64)
            ra, dec, cs_debiased, el_debiased, p_merge, p_edge, p_acw, p_cw, p_disk, petroRad_g, petroRad_i, petroRad_r = np.loadtxt(
                f+'.csv', skiprows=2, usecols=[1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], delimiter=',', unpack=True)
            self.objid[bf] = objid
            self.spiral[bf] = spiral
            self.elliptical[bf] = elliptical
            self.uncertain[bf] = uncertain
            self.nvote[bf] = nvote
            self.ra[bf] = ra
            self.dec[bf] = dec
            self.cs_debiased[bf] = cs_debiased
            self.el_debiased[bf] = el_debiased
            self.p_merge[bf] = p_merge
            self.p_edge[bf] = p_edge
            self.p_acw[bf] = p_acw
            self.p_cw[bf] = p_cw
            self.p_disk[bf] = p_disk
    def find_by_id(self, objid, g_type):
        bf = self.base_files[g_type]
        idx = np.argmax(objid==self.objid[bf])
        return (self.spiral[bf][idx],
            self.elliptical[bf][idx],
            self.uncertain[bf][idx],
            self.nvote[bf][idx],
            self.ra[bf][idx],
            self.dec[bf][idx],
            self.cs_debiased[bf][idx],
            self.el_debiased[bf][idx],
            self.p_merge[bf][idx],
            self.p_edge[bf][idx],
            self.p_acw[bf][idx],
            self.p_cw[bf][idx],
            self.p_disk[bf][idx])
    
    def find_by_type(self, g_type):
        return (self.objid[bf],
            self.spiral[bf],
            self.elliptical[bf],
            self.uncertain[bf],
            self.nvote[bf],
            self.ra[bf],
            self.dec[bf],
            self.cs_debiased[bf],
            self.el_debiased[bf],
            self.p_merge[bf],
            self.p_edge[bf],
            self.p_acw[bf],
            self.p_cw[bf],
            self.p_disk[bf])