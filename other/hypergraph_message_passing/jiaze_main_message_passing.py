import logging
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sem.str_to_type import none_or_type
from .src.data.data_io import load_data
from .src.model import dynamic_updates
from .src.model.jiaze_hyper_block_model import HyperBlockModel


class Arguments:
    def __init__(self):
        # Data IO.
        self.real_dataset = None  # Name of a real dataset to be loaded.
        """Path to a file containing a list of hyperedges representing a hypergraph."""
        self.hye_file = "./data/synthetic/sample.pkl"
        self.pickle_file = None  # Path to a file containing a pickle serialized hypergraph.
        self.save_dir = Path("./mp_results")  # Directory where results are saved.
        # Data parameters.
        """The maximum hyperedge size considered. This value is used to exclude 
        hyperedges in the configurations hypergraph, as well as a parameter of 
        the probabilistic model to compute internal quantities. """
        self.max_hye_size = None
        self.hye_sizes = None  # all possible order of hyperedges
        # Model parameters.
        # Number of nodes in the configurations hypergraph. Only needed (optionally) when specifying hye_file.
        self.N = None
        self.K = 4  # Number of communities in the model.
        """ Prior parameters for the communities of the stochastic block model. 
        This is a path to a file to be opened via numpy.loadtxt or numpy.load(our saved parameter). 
        If not provided, the value of n is initialized at random. """
        self.n = "./data/synthetic/n_prior.txt"
        """tensor dict of community interaction probabilities.
        This is a path to a file to be opened via numpy.loadtxt or numpy.load(our saved parameter).
        If not provided, the value of p is initialized at random. """
        self.p = "./data/synthetic/p_phase_transition_1.txt"
        # Model training.
        """Train with different various random initializations and
        choose only the model attaining the best log-likelihood."""
        self.train_rounds = 1
        self.em_iter = 1  # Max iterations of the EM procedure.
        """Threshold for the parameter change during EM. 
        The difference is computed with respect to the affinity matrix p and the community prior n."""
        self.em_thresh = 1.0e-5
        self.mp_iter = 2000  # Max iterations of the message passing procedure.
        """Threshold for the parameter change during message passing. 
        The difference is computed with respect to the log-marginal values."""
        self.mp_thresh = 1.0e-5
        """Number of consecutive steps where the change in log-marginals is below the mp_thresh 
        before message passing is stopped."""
        self.mp_patience = 50
        self.dirichlet_init_alpha = None  # Dirichlet alpha utilized for the model initialization.
        self.dropout = 0.75  # Dropout in the message passing updates.
        """Maximum number of parallel jobs. 
        1 means no parallelization, -1 means all the available cores."""
        self.n_jobs = -1
        self.seed = 123  # Random seed.
        self.logging = "INFO"  # Logging level.

    def __str__(self):
        result = ""
        attrs = vars(self)
        for attr, value in attrs.items():
            result += f'{attr} {value}\n'
        return result


def main0(args):
    random.seed(args.seed)
    logging.getLogger().setLevel(args.logging.upper())

    hyg = load_data(
        args.real_dataset,
        args.hye_file,
        args.pickle_file,
        args.N,
    )
    if args.max_hye_size is not None:
        hyg = hyg.max_hye_size_select(args.max_hye_size)

    if args.n is not None:
        if args.n.endswith('txt'):
            n = np.loadtxt(args.n)
        elif args.n.endswith('npz'):
            n = np.load(args.n, allow_pickle=True)['n_prior']
    else:
        n = None

    if args.p is not None:
        if args.n.endswith('txt'):
            p = np.loadtxt(args.p)
        elif args.n.endswith('npz'):
            p = np.load(args.n, allow_pickle=True)['ps_prior'].item()
    else:
        p = None

    # Set maximum number of parallel jobs.
    dynamic_updates.N_JOBS = args.n_jobs

    best_model = None
    best_free_energy = float("inf")
    all_free_energy = []
    for i in range(args.train_rounds):
        model = HyperBlockModel(
            n=n, p=p, N=hyg.N, K=args.K, max_hye_size=hyg.max_hye_size, hye_sizes=args.hye_sizes
        )
        model.em_inference(
            hypergraph=hyg,
            em_iter=args.em_iter,
            em_thresh=args.em_thresh,
            mp_iter=args.mp_iter,
            mp_thresh=args.mp_thresh,
            mp_patience=args.mp_patience,
            seed=args.seed + i * 1024 if args.seed is not None else None,
            dirichlet_alpha=args.dirichlet_init_alpha,
            dropout=args.dropout,
        )
        free_energy = model.free_energy(hyg)
        all_free_energy.append(free_energy)
        if free_energy < best_free_energy:
            best_model = model
            best_free_energy = free_energy

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

        # Save arguments.
        with open(args.save_dir / "args.txt", "w") as file:
            file.write(str(args))

        # Inference results.
        np.savez(
            args.save_dir / "inferred_params.npz",
            log_marginals=best_model.log_marginals,
            log_hye_to_node=best_model.log_hye_to_node,
            log_node_to_hye=best_model.log_node_to_hye,
            external_field=best_model.external_field,
            best_free_energy=best_free_energy,
            all_free_energy=all_free_energy,
            p=best_model.p,
            n=best_model.n,
            n_diff=best_model.n_diff,
            c_diff=best_model.c_diff,
            log_marginal_diff=np.hstack(best_model.log_marginal_diff),
            mp_iter_per_em_iter=[len(x) for x in best_model.log_marginal_diff],
        )


if __name__ == "__main__":
    arg = Arguments()
    arg.hye_file = "./data/jiaze_synthetic/test_sample.txt"
    arg.n = "./data/jiaze_synthetic/test_parameter.npz"
    arg.p = "./data/jiaze_synthetic/test_parameter.npz"
    arg.K = 2
    arg.hye_sizes = [2, 3]
    arg.save_dir = Path("./mp_results/jiaze")
    arg.dropout = 0
    main0(arg)

