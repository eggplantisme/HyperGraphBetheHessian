"""
Utilities for the implementation of the message passing dynamic programming updates.
"""
from __future__ import annotations

import itertools
from tqdm import tqdm

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse, special

from ..data.representation.incidence_hypergraph import IncidenceHypergraph

# Number of jobs utilized by joblib.Parallel, see
# https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
# This parallelizes the message computations over the hyperedges.
N_JOBS: int = -1


def compute_eta_dynamic_programming(
    hypergraph: IncidenceHypergraph,
    model: "HyperBlockModel",
    mask: np.ndarray | None = None,
) -> list[sparse.coo_array]:
    """Compute the eta quantities via dynamic programming.

    Parameters
    ----------
    hypergraph: configurations hypergraph.
    model: configurations stochastic block model.
    mask: a boolean mask to compute the psi values only for specific (hyperedge, node)
        pairs.
        The mask needs to be a flattened boolean array with the same length as
        hypergraphs.get_binary_incidence_matrix().data

    Returns
    -------
    The eta values, results of the dynamic programming recursions.
    """
    # The incidence matrix needs to be in CSC sparse format, the rest of the code
    # doesn't work otherwise.
    incidence: sparse.csc_array = hypergraph.get_binary_incidence_matrix()
    assert isinstance(incidence, sparse.csc_array), "Incidence matrix is not CSC."
    # For coherence with the returned COO array at the end, the incidence matrix needs
    # to be in canonical sorted format. Otherwise, calling all_psi.tocsc() might result
    # in a matrix where non-zero indices do not correspond.
    assert incidence.has_sorted_indices, (
        "The incidence matrix doesn't have a canonical sorted format. "
        "To fix this, call the sort_indices() method of scipy CSC matrices.",
    )
    if mask is not None:
        assert mask.shape == (len(incidence.data),), (
            f"The mask has shape {mask.shape}, "
            f"different from the incidence matrix data {incidence.data.shape}"
        )

    log_node_to_hye = [x.tocsc() for x in model.log_node_to_hye]
    K = model.K

    def hyperedge_eta_(hye: int):
        nodes, psi = hyperedge_eta(
            incidence,
            hye,
            model.cin,
            model.cout,
            K,
            log_node_to_hye,
            eta_tilde=False,
            mask=mask,
        )
        return hye, nodes, psi

    # res = Parallel(n_jobs=N_JOBS, verbose=10)(
    #     delayed(hyperedge_psi_)(hye) for hye in range(hypergraph.E)
    # )
    res = [hyperedge_eta_(hye) for hye in tqdm(range(hypergraph.E), desc="calculating eta...")]

    nonzeros = mask.sum() if mask is not None else incidence.nnz
    hye_idx = np.zeros(nonzeros)
    node_idx = np.zeros(nonzeros)
    psi_vals = np.zeros((nonzeros, K))

    idx = itertools.count()
    for hye, nodes, psi in res:
        for i, node in enumerate(nodes):
            idx_ = next(idx)
            hye_idx[idx_] = hye
            node_idx[idx_] = node
            psi_vals[idx_, :] = psi[i, :]

    all_psi = [
        sparse.coo_array(
            (psi_vals[:, a], (node_idx, hye_idx)),
            shape=(hypergraph.N, hypergraph.E),
        )
        for a in range(K)
    ]

    return all_psi


def compute_eta_tilde_dynamic_programming(
    hypergraph: IncidenceHypergraph,
    model: "HypergraphBlockModel",
) -> np.ndarray:
    """Compute the eta tilde quantities via dynamic programming. Used in free energy calculation

    Parameters
    ----------
    hypergraph: configurations hypergraph.
    model: configurations stochastic block model.

    Returns
    -------
    The psi tilde values, results of the dynamic programming recursions.
    """
    # Here we are assuming the incidence to be a CSC sparse array, the rest of the code
    # doesn't work otherwise.
    incidence: sparse.csc_array = hypergraph.get_binary_incidence_matrix()
    assert isinstance(
        incidence, sparse.csc_array
    ), "Incidence matrix is not is CSC sparse format."
    log_node_to_hye = [x.tocsc() for x in model.log_node_to_hye]

    def hyperedge_eta_tilde_(hye):
        eta_tilde = hyperedge_eta(
            incidence,
            hye,
            model.cin,
            model.cout,
            model.K,
            log_node_to_hye,
            eta_tilde=True
        )
        return hye, eta_tilde

    # res = Parallel(n_jobs=N_JOBS)(
    #     delayed(hyperedge_eta_)(hye) for hye in range(hypergraph.E)
    # )
    res = [hyperedge_eta_tilde_(hye) for hye in tqdm(range(hypergraph.E), desc="calculating eta...")]

    all_eta = np.zeros(hypergraph.E)
    for hye, eta_val in res:
        all_eta[hye] = eta_val

    return all_eta


def hyperedge_eta(
    incidence: sparse.csc_array,
    hye: int,
    cin: float,
    cout: float,
    K: int,
    log_node_to_hye: list[sparse.csc_array],
    eta_tilde: bool = False,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Compute the eta values for hyperedge indexed by hye
    Parameters
    ----------
    incidence: incidence matrix.
         The matrix, of shape (N, E), where N is the number of nodes and E of
         hyperedges, contains ones describing the nodes belonging to each hyperedge.
    hye: hyperedge index. The psi value is computed for this hyperedge.
    cin:
    cout:
    K: number of community
    log_node_to_hye: log-messages from nodes to hyperedges
    eta_tilde: whether to compute the eta or eta tilde value (respectively used for the
        message updates and partition function computations)
    mask: only perform the computation of eta for the specific nodes inside the
        hyperedge specified by the mask.

    Returns
    -------
    The psi values. If no mask is specified, also the node indices of the hyperedge are
    returned.
    """
    if mask is not None and eta_tilde:
        raise ValueError("Cannot provide a node mask when eta_tilde=True.")

    # The following declaration of the variable `nodes` is equivalent to selecting a
    # column as:
    # `nodes = incidence.getcol(hye).indices`
    # but much more efficient for CSC sparse arrays.
    nodes = incidence.indices[incidence.indptr[hye] : incidence.indptr[hye + 1]]

    messages = np.zeros((len(nodes), K))
    for a in range(K):
        # Similar to above, the following is equivalent to, but more efficient than:
        # messages[:, a] = log_node_to_hye[a].getcol(hye).data
        message = log_node_to_hye[a]
        messages[:, a] = message.data[message.indptr[hye] : message.indptr[hye + 1]]
    assert messages.shape == (len(nodes), K)

    if eta_tilde:
        return _compute_eta(cin, cout, messages, eta_tilde=True)

    if mask is not None:
        node_mask = mask[incidence.indptr[hye] : incidence.indptr[hye + 1]]
        nodes_to_process = node_mask.sum()
    else:
        node_mask = (True for _ in range(len(nodes)))
        nodes_to_process = len(nodes)

    eta_vals = []
    for i, (node, mask_val) in enumerate(zip(nodes, node_mask)):
        if mask_val:
            # All messages but that to the node.
            partial_message = np.delete(messages, i, axis=0)
            assert partial_message.shape == (len(nodes) - 1, K)
            eta = _compute_eta(cin, cout, partial_message, eta_tilde=False)
            eta_vals.append(eta)

    if eta_vals:
        eta_vals = np.stack(eta_vals)
        assert eta_vals.shape == (nodes_to_process, K)
    else:
        eta_vals = np.array([])

    if mask is not None:
        return nodes[node_mask], eta_vals
    return nodes, eta_vals


def _compute_eta(
    cin, cout,
    log_node_to_hye_array: np.ndarray,
    eta_tilde: bool = False,
) -> np.ndarray:
    """Compute the eta dynamic programming recursions to obtain the psi values.
    This function computes the array of eta values for a single (hyperedge, node) pair.

    Parameters
    ----------
    cin:
    cout:
    log_node_to_hye_array: an array containing the log-messages for the nodes in the
        hyperedge.
        If the hyperedge has size f and there are K communities, this array has shape
        (f-1, K), containing the log-messages from the hyperedge to all the nodes except
        the one for which eta is computed.
        If eta_tilde=True, log_node_to_hye_array instead needs to be the array of shape
        (f, K), containing all log-messages from the hyperedge to all nodes.
    eta_tilde: compute the value of eta tilde, appearing in the computations of the
        log-likelihood, as opposed to the value eta utilized during message passing.

    Returns
    -------
    The K-dimensional array of eta(f, i, t_i) values, for fixed hyperedge f and node i
    in f, and for all the possible values t_i=1, ..., K.
    """
    K = np.shape(log_node_to_hye_array)[1]
    if eta_tilde is False:
        f = np.shape(log_node_to_hye_array)[0] + 1  # f is the size of the hyperedge
    else:
        f = np.shape(log_node_to_hye_array)[0]  # f is the size of the hyperedge
    if eta_tilde:
        eta_t = cin[f]
        for idx in range(log_node_to_hye_array.shape[0]):
            if idx == 0:
                continue
            else:
                for psi_idx in range(log_node_to_hye_array.shape[1]):
                    select_log_node_to_hye_array = np.delete(log_node_to_hye_array[:idx, :], psi_idx, axis=1)
                    exp_col_sum_but_psi_idx = np.exp(np.sum(select_log_node_to_hye_array, axis=0))
                    sum_exp_col_sum_but_psi_idx = np.sum(exp_col_sum_but_psi_idx)
                    eta_t += (cout[f] - cin[f]) * np.exp(log_node_to_hye_array[idx, psi_idx]) * sum_exp_col_sum_but_psi_idx
        if eta_t<0:
            # exp_eta can be negative near zero due to numerical issues, which can cause NaN values in the log. 
            # We add a small value to avoid this issue.
            eta_t = 1e-10
        return eta_t
    else:
        exp_eta = np.ones(K) * cin[f]
        row_sum_exp = np.sum(np.exp(log_node_to_hye_array), axis=1)
        for idx in range(log_node_to_hye_array.shape[0]):
            if idx == 0:
                exp_col_sum = 1
            else:
                select_log_node_to_hye_array = log_node_to_hye_array[:idx, :]
                exp_col_sum = np.exp(np.sum(select_log_node_to_hye_array, axis=0))
            exp_eta = row_sum_exp[idx] * exp_eta + (row_sum_exp[idx] - np.exp(log_node_to_hye_array[idx, :])) * (cout[f] - cin[f]) * exp_col_sum
        if np.any(exp_eta<0):
            # exp_eta can be negative near zero due to numerical issues, which can cause NaN values in the log. 
            # We add a small value to avoid this issue.
            exp_eta[exp_eta<0] = 1e-10 
        eta = np.log(exp_eta)
        if np.any(np.isnan(eta)):
            raise ValueError(f"NaN values in eta, exp_eta: {exp_eta}, log_node_to_hye_array: {log_node_to_hye_array}")
        return eta


########################################################################################
# The following functions are slow, and should only be used for testing.
def _compute_psi_brute_force(
    hypergraph: IncidenceHypergraph,
    model: "HypergraphBlockModel",
) -> list[sparse.coo_array]:
    """Compute the psi values only by their definition.
    This function is computationally slow and should only be utilized for testing.

    Parameters
    ----------
    hypergraph: configurations hypergraph.
    model: configurations stochastic block model.

    Returns
    -------
    The psi values.
    """
    # Here we are assuming the incidence to be a CSC sparse array, the rest of the code
    # doesn't work otherwise.
    incidence: sparse.csc_array = hypergraph.get_binary_incidence_matrix()
    hye_idx = [[] for _ in range(model.K)]
    node_idx = [[] for _ in range(model.K)]
    message_val = [[] for _ in range(model.K)]
    for hye in range(hypergraph.E):
        nodes = incidence.getcol(hye).indices
        partial_node_to_hye = np.stack(
            [model.log_node_to_hye[a][nodes, [hye]] for a in range(model.K)],
            axis=1,
        )
        assert partial_node_to_hye.shape == (len(nodes), model.K)

        all_idx = np.arange(len(nodes))
        for idx, node in enumerate(nodes):
            for node_assignment in range(model.K):
                # Possible combinations of node assignments: all node assignments
                # vary from 0 to K-1, apart from the node we are computing the
                # message for, which takes a fixed node_assigment.
                all_combinations = itertools.product(
                    *(
                        range(model.K) if i != idx else (node_assignment,)
                        for i in range(len(nodes))
                    )
                )
                pi_vals = []
                message_vals = []
                for assign in all_combinations:
                    assert len(assign) == len(nodes)
                    hye_pi = model.single_hye_pi(assign)
                    pi_vals.append(hye_pi)
                    messages = np.delete(partial_node_to_hye[all_idx, assign], idx)
                    assert messages.shape == (len(nodes) - 1,)
                    message_vals.append(messages.sum())

                # Weighted log-sum-exp operation.
                assert len(pi_vals) == len(message_vals)
                final_message = special.logsumexp(message_vals, b=pi_vals)
                # Store indices for COO matrices to be constructed in the end.
                hye_idx[node_assignment].append(hye)
                node_idx[node_assignment].append(node)
                message_val[node_assignment].append(final_message)

    psi = [
        sparse.coo_array(
            (
                message_val[a],
                (node_idx[a], hye_idx[a]),
            ),
            shape=(hypergraph.N, hypergraph.E),
        )
        for a in range(model.K)
    ]
    assert all(new_mat.shape == incidence.shape for new_mat in psi)
    assert all(not np.any(((new_mat != 0) != incidence).data) for new_mat in psi)

    return psi


def _compute_eta_brute_force(
    model, p: np.ndarray, log_node_to_hye_array: np.ndarray
) -> np.ndarray:
    """Compute the eta values only by their definition.
    This function is computationally slow and should only be utilized for testing.

    Parameters
    ----------
    model: configurations stochastic block model.
    p: symmetric matrix of inter and intra-community probabilities.
    log_node_to_hye_array: the logarithm of the messages from nodes to hyperedges.

    Returns
    -------
    The eta values.
    """
    K = p.shape[0]
    n_other_nodes = log_node_to_hye_array.shape[0]

    eta_vals = []
    for node_assignment in range(K):
        pi_vals = []
        log_message_sums = []
        for other_assignments in itertools.product(
            *(range(K) for _ in range(n_other_nodes))
        ):
            all_assignments = other_assignments + (node_assignment,)
            pi_vals.append(model.single_hye_pi(all_assignments))
            log_message_sums.append(
                log_node_to_hye_array[
                    np.arange(len(other_assignments)), other_assignments
                ].sum()
            )
        assert len(log_message_sums) == len(pi_vals) == K ** n_other_nodes
        eta_vals.append(special.logsumexp(log_message_sums, b=pi_vals))

    assert len(eta_vals) == K
    return np.array(eta_vals)
