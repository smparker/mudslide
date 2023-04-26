import os
import numpy as np
import json
from mudslide.electronics import ElectronicModel_
from typing import Tuple, Any
from mudslide.periodic_table import masses
from mudslide.constants import eVtoHartree, amu_to_au
from nff.md.tully.io import get_results, coords_to_xyz
from nff.train import load_model
from mudslide.typing import ArrayLike, DtypeLike
from mudslide.batch import TrajGenConst, TrajGenNormal, BatchedTraj
import copy as cp

class NFFModel(ElectronicModel_):

    def __init__(
        self,
        states: ArrayLike,
        model_path: str,
        coord_path: str,
        nbrs: int,
        cutoff:int,
        cutoff_skin:float,
        device:str,
        batch_size: int,
        num_states: int,
        surf: int,
        max_gap_hop:int,
        decoherence_type: bool,
        needs_nacv: bool,
        all_engrads: bool,
        needs_nbrs: bool = True, 
        representation: str = "adiabatic",
        reference: Any = None
        ):

        self.states = states
        self.nstates_ = len(self.states)

        self.model_path =  model_path
        self.coord_path = coord_path
        self.nbrs = nbrs 
        self.cutoff = cutoff 
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.batch_size = batch_size
        self.num_states = num_states
        self.surf = surf
        self.max_gap_hop = max_gap_hop
        self.needs_nacv = needs_nacv
        self.all_engrads = all_engrads
        self.needs_nbrs = needs_nbrs

        # always need couplings. 
        self.nacv = True
        self.decoherence_type = decoherence_type 
        self.all_engrads = 'subotnik' in self.decoherence_type
        
        self.all_params = {}

        self.nff_init()
        

        ElectronicModel_.__init__(self, representation=representation, reference=reference)


    def nstates(self):
        return self.nstates_

    def ndim(self):
        return self.ndim_

    def nff_init(self):
        self.get_coords()
        self.all_params["model_path"] = self.model_path
        nxyz = [coords_to_xyz(self.all_params["coords"])]
        self.all_params["nxyz"] = nxyz
        self.num_atoms = len(nxyz)
        self.old_U = None

        with open(os.path.join(self.all_params["model_path"], "params.json"), 'r') as f:
            self.model_params = json.load(f) 
        self.model = load_model(self.all_params["model_path"], self.model_params, self.model_params['model_type'])  

        self.update_props()
        # now we reassign self.old_U to whatever comes out of update_props

    def get_coords(self):
        with open(self.coord_path, 'r') as f:
            coords = json.load(f)
        self.all_params["coords"] = coords
        coords_list = []
        self.mass = []
        
        for c in coords:
            coords_list.append([c["x"], c["y"],c["z"]])
            self.mass.append(3 * [masses[c["element"].lower()]])

        self.ndim_ = 3 * len(coords)
        self.X = np.array(coords_list, dtype=np.float64).reshape(self.ndim())


        self.mass = np.array(self.mass, dtype=np.float64).reshape(self.ndim()) * amu_to_au


    def update_coords(self, X):
        X = X.reshape((self.ndim() // 3, 3))

        for i, coord in enumerate (X):
            self.all_params["coords"][i]["x"] = coord[0]
            self.all_params["coords"][i]["y"] = coord[1]
            self.all_params["coords"][i]["z"] = coord[2]
            
        with open (self.coord_path, "w") as f: 
            json.dump(self.all_params["coords"], f, indent = 3)
        nxyz = [coords_to_xyz(self.all_params["coords"])]
        self.all_params["nxyz"] = nxyz

    def update_props(self):

        props = get_results(model=self.model,
                            nxyz=self.all_params["nxyz"],
                           # nbr_list=nbrs,
                            nbr_list=self.nbrs,
                            num_atoms=self.num_atoms,
                            needs_nbrs=self.needs_nbrs,
                            cutoff=self.cutoff,
                            cutoff_skin=self.cutoff_skin,
                            device=self.device,
                            batch_size=self.batch_size,
                            old_U=self.old_U,
                            num_states=self.num_states,
                            surf=self.surf,
                            max_gap_hop=self.max_gap_hop,
                            all_engrads=self.all_engrads,
                            nacv=self.needs_nacv,
                            diabat_keys = self.model_params["details"]["diabat_keys"]
                            )
        self.props = props

        self.derivative_coupling = np.zeros((self.nstates(), self.nstates(), self.ndim()))
        for i in range(self.nstates()):
            for j in range(i+1, self.nstates()):
                self.derivative_coupling[i][j] = self.props[f"nacv_{i}{j}"].reshape(self.ndim()) 
                self.derivative_coupling[j][i] = -(self.derivative_coupling[i][j])

        self.gradients = []
        for i in range (self.nstates()):
            self.gradients.append(self.props[f"d_{i}{i}_grad"].reshape(self.ndim()))
        
        self.gradients = np.array(self.gradients)

        self.force = -self.gradients 
    
    def compute(self, X, couplings, gradients, reference):
        """
        xxxxx
        """
        self.update_props()

        self.energies = [self.props[f"energy_{i}"][0,0] for i in range (self.nstates())] 
        self.hamiltonian = np.zeros([self.nstates(), self.nstates()])
        for i, e in enumerate(self.states):
            self.hamiltonian[i][i] = self.energies[e]
        return self.hamiltonian

    def update(self, X: ArrayLike, electronics: Any=None, couplings: Any = None, gradients: Any = None):
        out = cp.copy(self)
        out.position = X 
        out.update_coords(X)
        out.compute(X, couplings=couplings, gradients=gradients, reference=self.reference)
        return out 

    def clone(self):
        model_clone = cp.deepcopy(self)
        model_clone.nff_init()
        return model_clone


def main():
    run()

if __name__ == "__main__":
    main()
