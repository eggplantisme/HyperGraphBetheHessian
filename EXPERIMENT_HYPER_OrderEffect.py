import time
import os
import numpy as np
import itertools
from sklearn.metrics.cluster import adjusted_mutual_info_score
from _HyperSBM import *
from _CommunityDetect import CommunityDetect
from _HyperCommunityDetection import HyperCommunityDetect
from _FigureJiazeHelper import get_confusionmatrix
from multiprocessing import Pool

BP_mp_patience = None
BP_mp_thresh = None

def CDwithBH(hsbm, given_q, projection=False, only_assortative=False):
    start = time.time()
    if projection == "Clique":
        A = hsbm.H.dot(hsbm.H.T) - diags(hsbm.H.dot(hsbm.H.T).diagonal())
        A[A.nonzero()] = 1
        BH_Partition, BH_NumGroup = CommunityDetect(A).BetheHessian(num_groups=given_q, only_assortative=only_assortative)
    else:
        BH_Partition, BH_NumGroup = HyperCommunityDetect().BetheHessian(hsbm, num_groups=given_q, only_assortative=only_assortative)
    return BH_Partition
    # cd_time = time.time() - start
    # cm, _ = get_confusionmatrix(hsbm.groupId, BH_Partition, hsbm.q, BH_NumGroup)
    # ami = adjusted_mutual_info_score(hsbm.groupId, BH_Partition)
    # print(f"BH result AMI: {ami}. Time={cd_time}. Confusion Matrix({np.shape(cm)}) is: \n{cm}")
    # return ami, BH_NumGroup, cd_time

def CDwithBP(hsbm, arg=None, hypergraph_save_name=None, parameters=None):
    start = time.time()
    if arg is None:
        arg = dict()
        arg["q"] = 2
        arg["hyperedge_sizes"] = hsbm.Ks
        path = "./other/hypergraph_message_passing/data/jiaze_synthetic/"
        if hypergraph_save_name is None:
            arg["hypergraph"] = path + f'_n={hsbm.n}_q={hsbm.q}_Ks={hsbm.Ks}_hgraph.txt'
            arg["hsbm_parameter"] = path + f'_n={hsbm.n}_q={hsbm.q}_Ks={hsbm.Ks}_parameter.npz'
            arg["save_dir"] = path + f'_n={hsbm.n}_q={hsbm.q}_Ks={hsbm.Ks}_bpresult'
        else:
            arg["hypergraph"] = path + f'{hypergraph_save_name}_hgraph.txt'
            arg["hsbm_parameter"] = path + f'{hypergraph_save_name}_parameter.npz'
            arg["save_dir"] = path + f'{hypergraph_save_name}_bpresult'
    if parameters is not None:
        np.savez(arg["hsbm_parameter"], n_prior=parameters['n_prior'], ps_prior=parameters['ps_prior'])
    else:
        print("Aggregated parameter not provided!")
        return None, None, None, None
    arg['mp_patience'] = 10 if BP_mp_patience is None else BP_mp_patience
    arg["mp_thresh"] = 1e-3 if BP_mp_thresh is None else BP_mp_thresh
    BP_Partition, BP_NumGroup, BP_FreeEnergy = HyperCommunityDetect().BeliefPropagation(hsbm, arg, return_free_energy=True, parameter_saved=True)
    cd_time = time.time() - start
    cm, _ = get_confusionmatrix(hsbm.groupId, BP_Partition, hsbm.q, BP_NumGroup)
    # ami = adjusted_mutual_info_score(hsbm.groupId, BP_Partition)
    print(f"BP result Time={cd_time}. Confusion Matrix({np.shape(cm)}) is: \n{cm}")
    return BP_Partition, BP_NumGroup, cd_time, BP_FreeEnergy

def exp_subprocess(n=4000, q=4, d=10, epsilon_star=1, Ks=(2, 3, ), times=1, save_path=None, only_assortative=False, old=False, diff_shape=False, method="BH"):
    sizes = [int(n / q)] * q
    ps_dict = dict()
    aggregated_parameters = dict({"n_prior": None, "ps_prior": None})
    aggregated_parameters["n_prior"] = np.array([int(n/q)*2]*2) / n
    if old:
        a = q**2*d*epsilon_star / (2*(epsilon_star + 1))
        b = n * a / (epsilon_star * (n/q-1))
        a = a / n**(1)
        b = b / n**(2)
        ps_dict[2] = np.array(
            [[0, 0, a, 0], 
            [0, 0, 0, a], 
            [a, 0, 0, 0], 
            [0, a, 0, 0]]
        )  # order 2 hyperedge only exist between (0, 2), (1, 3)
        ps_dict[3] = np.zeros(tuple([q]*3))
        for index in itertools.product(range(q), repeat=3):
            if (0 in index and 1 in index and 2 not in index and 3 not in index) or (2 in index and 3 in index and 0 not in index and 1 not in index):
                ps_dict[3][index] = b  # order 3 hyperedge only exist between (0, 1), (2, 3)
    elif Ks[0] == 4 and Ks[1]==4 and diff_shape:
        b = 2*4**3*d/(epsilon_star+1)  # 3*4**4 *d / (3*epsilon_star + 1)
        a = 6*4**2*d*epsilon_star/(epsilon_star+1)  # 3*b*epsilon_star / 2
        a = a / n**(Ks[0]-1)
        b = b / n**(Ks[1]-1)
        ps_dict[4] = np.zeros(tuple([q]*4))
        for index in itertools.product(range(q), repeat=4):
            gid, count = np.unique(index, return_counts=True)
            gid_count = dict(zip(gid, count))
            if 0 in gid_count.keys() and gid_count[0] == 2 and 1 in gid_count.keys() and gid_count[1] == 2:
                ps_dict[4][index] = b
            if 2 in gid_count.keys() and gid_count[2] == 2 and 3 in gid_count.keys() and gid_count[3] == 2:
                ps_dict[4][index] = b
            if 0 in gid_count.keys() and gid_count[0] == 1 and 2 in gid_count.keys() and gid_count[2] == 3:
                ps_dict[4][index] = a
            if 0 in gid_count.keys() and gid_count[0] == 3 and 2 in gid_count.keys() and gid_count[2] == 1:
                ps_dict[4][index] = a
            if 1 in gid_count.keys() and gid_count[1] == 1 and 3 in gid_count.keys() and gid_count[3] == 3:
                ps_dict[4][index] = a
            if 1 in gid_count.keys() and gid_count[1] == 3 and 3 in gid_count.keys() and gid_count[3] == 1:
                ps_dict[4][index] = a
    elif Ks[0] == 5 and Ks[1] == 5 and diff_shape:
        b = 12*4**4*d/(5*(epsilon_star+1))  # 3*4**4 *d / (3*epsilon_star + 1)
        a = 24*4**4*d*epsilon_star/(5*(epsilon_star+1))  # 3*b*epsilon_star / 2
        a = a / n**(Ks[0]-1)
        b = b / n**(Ks[1]-1)
        ps_dict[5] = np.zeros(tuple([q]*5))
        for index in itertools.product(range(q), repeat=5):
            gid, count = np.unique(index, return_counts=True)
            gid_count = dict(zip(gid, count))
            if 0 in gid_count.keys() and gid_count[0] == 2 and 1 in gid_count.keys() and gid_count[1] == 3:
                ps_dict[5][index] = b
            if 0 in gid_count.keys() and gid_count[0] == 3 and 1 in gid_count.keys() and gid_count[1] == 2:
                ps_dict[5][index] = b
            if 2 in gid_count.keys() and gid_count[2] == 2 and 3 in gid_count.keys() and gid_count[3] == 3:
                ps_dict[5][index] = b
            if 2 in gid_count.keys() and gid_count[2] == 3 and 3 in gid_count.keys() and gid_count[3] == 2:
                ps_dict[5][index] = b
            if 0 in gid_count.keys() and gid_count[0] == 1 and 2 in gid_count.keys() and gid_count[2] == 4:
                ps_dict[5][index] = a
            if 0 in gid_count.keys() and gid_count[0] == 4 and 2 in gid_count.keys() and gid_count[2] == 1:
                ps_dict[5][index] = a
            if 1 in gid_count.keys() and gid_count[1] == 1 and 3 in gid_count.keys() and gid_count[3] == 4:
                ps_dict[5][index] = a
            if 1 in gid_count.keys() and gid_count[1] == 4 and 3 in gid_count.keys() and gid_count[3] == 1:
                ps_dict[5][index] = a
    else:
        a = q**Ks[0]*math.factorial(Ks[0])*d*epsilon_star / (2*(2**Ks[0]-2)*(Ks[0]*epsilon_star + Ks[1]))
        b = q**Ks[1]*math.factorial(Ks[1])*d / (2*(2**Ks[1]-2)*(Ks[0]*epsilon_star + Ks[1]))
        a = a / n**(Ks[0]-1)
        b = b / n**(Ks[1]-1)
        for i, k, c in zip([0, 1], [Ks[0], Ks[1]], [a, b]):
            if k not in ps_dict.keys():
                ps_dict[k] = np.zeros(tuple([q]*k))
            for index in itertools.product(range(q), repeat=k):
                if i == 0:
                    if (0 in index and 2 in index and 1 not in index and 3 not in index) or (1 in index and 3 in index and 0 not in index and 2 not in index):
                        ps_dict[k][index] = c  # order ks[0] hyperedge only exist between (0, 2), (1, 3)
                if i == 1:
                    if (0 in index and 1 in index and 2 not in index and 3 not in index) or (2 in index and 3 in index and 0 not in index and 1 not in index):
                        ps_dict[k][index] = c  # order ks[1] hyperedge only exist between (0, 1), (2, 3)
        
    results = ""
    for t in range(times):
        start = time.time()
        hsbm = HyperSBM(sizes, ps_dict)
        print(f'times={t} start. d={d}, E_{Ks[0]}/E_{Ks[1]}={epsilon_star}, hsbm construct time={time.time()-start}')
        # Community Detection
        if method == "BH":
            BH_Partition = CDwithBH(hsbm, given_q=2, only_assortative=only_assortative)  # Hyper BH
            group_01_23 = np.array([0] * (sizes[0]+sizes[1]) + [1] * (sizes[2]+sizes[3]))
            ami_01_23 = adjusted_mutual_info_score(group_01_23, BH_Partition)
            group_02_13 = np.array([0]*sizes[0] + [1]*sizes[1]+ [0]*sizes[2] + [1]*sizes[3])
            ami_02_13 = adjusted_mutual_info_score(group_02_13, BH_Partition)
            results += f'{epsilon_star} {t} {ami_01_23} {ami_02_13}'
            BH_Partition = CDwithBH(hsbm, given_q=2, projection="Clique", only_assortative=only_assortative)  # Network BH on Clique projection
            ami_01_23 = adjusted_mutual_info_score(group_01_23, BH_Partition)
            ami_02_13 = adjusted_mutual_info_score(group_02_13, BH_Partition)
            results += f' {ami_01_23} {ami_02_13}\n'
            print(f'times={t} end. d={d}, E_2/E_3={epsilon_star}, total time={time.time()-start}')
        elif method == "BP_01_23":
            aggregated_parameters["ps_prior"] = dict()
            if not diff_shape:
                for i, k, c in zip([0, 1], [Ks[0], Ks[1]], [a, b]):
                    if k not in aggregated_parameters["ps_prior"].keys():
                        aggregated_parameters["ps_prior"][k] = np.zeros(tuple([2]*k)) * 1e-10  # add small value to avoid zero probability which may cause numerical issue in BP
                    for index in itertools.product(range(2), repeat=k):
                        if i == 0:
                            if (0 in index and 1 in index):
                                aggregated_parameters["ps_prior"][k][index] = c/(2**(k-1))
                        if i == 1:
                            if (0 in index and 1 not in index) or (1 in index and 0 not in index):
                                aggregated_parameters["ps_prior"][k][index] = c*(2**k-2)/(2**k)
            hypergraph_save_name = f'OrderShapeEffect_exp_{method}_n={n}_d={d}_Ks={Ks}_epsilon={epsilon_star}_times={t}'
            result = CDwithBP(hsbm, hypergraph_save_name=hypergraph_save_name, parameters=aggregated_parameters)
            group_01_23 = np.array([0] * (sizes[0]+sizes[1]) + [1] * (sizes[2]+sizes[3]))
            ami_01_23 = adjusted_mutual_info_score(group_01_23, result[0])
            group_02_13 = np.array([0]*sizes[0] + [1]*sizes[1]+ [0]*sizes[2] + [1]*sizes[3])
            ami_02_13 = adjusted_mutual_info_score(group_02_13, result[0])
            results += f'{epsilon_star} {t} {ami_01_23} {ami_02_13} {result[3]} {result[2]}\n'
            print(f'times={t} end. d={d}, E_2/E_3={epsilon_star}, total time={time.time()-start}')
        elif method == "BP_02_13":
            aggregated_parameters["ps_prior"] = dict()
            if not diff_shape:
                for i, k, c in zip([0, 1], [Ks[0], Ks[1]], [a, b]):
                    if k not in aggregated_parameters["ps_prior"].keys():
                        aggregated_parameters["ps_prior"][k] = np.zeros(tuple([2]*k)) * 1e-10  # add small value to avoid zero probability which may cause numerical issue in BP
                    for index in itertools.product(range(2), repeat=k):
                        if i == 0:
                            if (0 in index and 1 not in index) or (1 in index and 0 not in index):
                                aggregated_parameters["ps_prior"][k][index] = c*(2**k-2)/(2**k)
                        if i == 1:
                            if (0 in index and 1 in index):
                                aggregated_parameters["ps_prior"][k][index] = c /(2**(k-1))
            hypergraph_save_name = f'OrderShapeEffect_exp_{method}_n={n}_d={d}_Ks={Ks}_epsilon={epsilon_star}_times={t}'
            result = CDwithBP(hsbm, hypergraph_save_name=hypergraph_save_name, parameters=aggregated_parameters)
            group_01_23 = np.array([0] * (sizes[0]+sizes[1]) + [1] * (sizes[2]+sizes[3]))
            ami_01_23 = adjusted_mutual_info_score(group_01_23, result[0])
            group_02_13 = np.array([0]*sizes[0] + [1]*sizes[1]+ [0]*sizes[2] + [1]*sizes[3])
            ami_02_13 = adjusted_mutual_info_score(group_02_13, result[0])
            results += f'{epsilon_star} {t} {ami_01_23} {ami_02_13} {result[3]} {result[2]}\n'
            print(f'times={t} end. d={d}, E_2/E_3={epsilon_star}, total time={time.time()-start}')
    return save_path, results

def write_results(arg):
    """
    :param arg: savepath, results
    :return:
    """
    if arg[0] is not None:
        with open(arg[0], 'a') as fw:
            fw.write(arg[1])


def print_error(value):
    print(value)


def run_exp(epsilon_stars, n, Ks, d, times, save_path=None, multiprocessing=True, only_assortative=False, diff_shape=False, method="BH"):
    epsilon_done = set()
    if save_path is not None and os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                epsilon_done.add(round(float(row[0]), 5))
    if multiprocessing:
        p = Pool(2)
        for epsilon in epsilon_stars:
            if round(epsilon, 5) in epsilon_done:
                print(f'epsilon_star={epsilon} has been run!')
                continue
            p.apply_async(exp_subprocess, args=(n, d, epsilon, Ks, times, save_path, only_assortative, diff_shape, method), callback=write_results, error_callback=print_error)
            pass
        p.close()
        p.join()
    else:
        for epsilon in epsilon_stars:
            if round(epsilon, 5) in epsilon_done:
                print(f'epsilon={epsilon} has been run!')
                continue
            savepath, results = exp_subprocess(n=n, d=d, epsilon_star=epsilon, Ks=Ks, times=times, save_path=save_path, only_assortative=only_assortative, diff_shape=diff_shape, method=method)
            write_results((savepath, results))

def read_exp(load_path, add_paths=None):
    """
    read the results file from run_exp
    :param load_path:
    :return:
    """
    with open(load_path, 'r') as f:
        results_list = [row.strip().split() for row in f.readlines()]
        if add_paths is not None:
            print("Additional result adding...")
            for add_path in add_paths:
                with open(add_path, 'r') as add_f:
                    results_list = results_list + [row.strip().split() for row in add_f.readlines()]
        results = np.round(np.float64(results_list), decimals=5)
        epsilons = np.unique(results[:, 0])
        result = np.zeros((np.size(epsilons), np.size(results[0]) - 1))
        i = 0
        for epsilon in epsilons:
            ami_results = results[np.squeeze(np.argwhere(results[:, 0] == epsilon))]
            if np.size(ami_results) == 0:
                print(f"Some parameter epsilon={epsilon} didn't run!")
            mean_ami = np.mean(ami_results, 0)
            result[i, :] = mean_ami[1:]
            i += 1
    return epsilons, result


def main0():
    epsilon_stars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
    run_exp(epsilon_stars, times=10, save_path="./result/hyperOrderEffect/exp8.14.txt", multiprocessing=False)


def main1():
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 10, 100))))
    run_exp(epsilon_stars, times=10, save_path="./result/hyperOrderEffect/exp8.14.txt", multiprocessing=False)

def main2():
    epsilon_stars = [10]
    run_exp(epsilon_stars, times=10, save_path=None, multiprocessing=False, only_assortative=False)

def main3():
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    run_exp(epsilon_stars, times=5, save_path="./result/hyperOrderEffect/exp8.15_onlyassort.txt", multiprocessing=False, only_assortative=True)

def main4():
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    run_exp(epsilon_stars, times=5, save_path="./result/hyperOrderEffect/exp8.18_onlyassort_amiwith2settinggroup.txt", multiprocessing=False, only_assortative=True)

def main5():
    n = 4000
    d = 10
    Ks = (2, 3, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.26_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main6():
    n = 8000
    d = 10
    Ks = (2, 3, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.26_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main7():
    n = 4000
    d = 10
    Ks = (2, 4, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.26_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main8():
    n = 8000
    d = 10
    Ks = (2, 4, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.26_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main9():
    n = 8000
    d = 10
    Ks = (2, 5, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.26_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main10():
    n = 8000
    d = 10
    Ks = (3, 4, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.26_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main11():
    n = 8000
    d = 10
    Ks = (4, 4, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.27_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_modify.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main12():
    n = 8000
    d = 10
    Ks = (4, 5, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp10.12_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main13():
    n = 8000
    d = 10
    Ks = (5, 5, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.26_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)


def main14():
    n = 30000
    d = 10
    Ks = (3, 4, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp8.26_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main15():
    n = 8000
    d = 10
    Ks = (3, 8, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp10.12_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)

def main16():
    n = 8000
    d = 10
    Ks = (4, 4, )
    diff_shape = False
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp10.12_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True, diff_shape=diff_shape)

def main17():
    n = 8000
    d = 50
    Ks = (2, 3, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp10.23_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)


def main18():
    n = 8000
    d = 50
    Ks = (2, 4, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp10.27_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)


def main19():
    n = 8000
    d = 50
    Ks = (2, 5, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp10.27_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True)


def main20():
    n = 8000
    d = 50
    Ks = (2, 3, )
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp10.28_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_allowdisassortative.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=False)


def main21():
    n = 8000
    d = 10
    Ks = (5, 5, )
    diff_shape = True
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    save_path = f"./result/hyperOrderEffect/exp11.3_onlyassort_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_diffshape.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True, diff_shape=diff_shape)


def main22():
    n = 8000
    d = 50
    Ks = (2, 3, )
    # epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 50), np.linspace(1, 4, 50), np.linspace(4, 5, 10))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    method = "BP_01_23"
    save_path = f"./result/hyperOrderEffect/exp26.03.26_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)

def main23():
    n = 8000
    d = 50
    Ks = (2, 3, )
    # epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 50), np.linspace(1, 4, 50), np.linspace(4, 5, 10))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    method = "BP_02_13"
    save_path = f"./result/hyperOrderEffect/exp26.03.30_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=2, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)

def main24():
    n = 8000
    d = 10
    Ks = (3, 4, )
    # epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 10), np.linspace(1, 5, 10))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    method = "BP_01_23"
    save_path = f"./result/hyperOrderEffect/exp26.04.05_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=5, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)

def main25():
    n = 8000
    d = 10
    Ks = (3, 4, )
    # epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 10), np.linspace(1, 5, 10))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    # These two parameter use on for this setting because BP is hard to converge (flipping often)
    BP_mp_patience = 5
    BP_mp_thresh = 0.01
    method = "BP_02_13"
    save_path = f"./result/hyperOrderEffect/exp26.04.05_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=2, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)


def main26():
    n = 8000
    d = 50
    Ks = (2, 3, )
    # epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    epsilon_stars = np.unique((np.linspace(4, 5, 10)))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    method = "BP_01_23"
    save_path = f"./result/hyperOrderEffect/exp26.04.06_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=10, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)
    # method = "BP_02_13"
    # save_path = f"./result/hyperOrderEffect/exp26.04.06_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    # run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=10, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)


def main27():
    n = 8000
    d = 10
    Ks = (3, 4, )
    # epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    epsilon_stars = np.unique(np.concatenate((np.linspace(5, 10, 10), np.linspace(10, 100, 10), np.linspace(100, 1000, 10))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    # These two parameter use on for this setting because BP is hard to converge (flipping often)
    BP_mp_patience = 5
    BP_mp_thresh = 0.01
    method = "BP_01_23"
    save_path = f"./result/hyperOrderEffect/exp26.04.07_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=2, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)
    method = "BP_02_13"
    save_path = f"./result/hyperOrderEffect/exp26.04.07_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=2, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)


def main28():
    n = 8000
    d = 10
    Ks = (4, 5, )
    # epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 10), np.linspace(5, 10, 10))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    # These two parameter use on for this setting because BP is hard to converge (flipping often)
    BP_mp_patience = 5
    BP_mp_thresh = 0.01
    method = "BP_01_23"
    save_path = f"./result/hyperOrderEffect/exp26.04.08_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=2, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)
    method = "BP_02_13"
    save_path = f"./result/hyperOrderEffect/exp26.04.08_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=2, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)


def main29():
    n = 8000
    d = 50
    Ks = (2, 4, )
    # epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 100), np.linspace(1, 2, 100), np.linspace(2, 10, 100))))
    epsilon_stars = np.unique(np.concatenate((np.linspace(0.1, 1, 10), np.linspace(1, 10, 10))))
    print(f"There are {np.size(epsilon_stars)} epsilon stars to run.")
    method = "BP_01_23"
    save_path = f"./result/hyperOrderEffect/exp26.04.09_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=10, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)
    method = "BP_02_13"
    save_path = f"./result/hyperOrderEffect/exp26.04.09_amiwith2settinggroup_n{n}_d{d}_k{Ks[0]}_kstar{Ks[1]}_{method}.txt"
    run_exp(epsilon_stars, n=n, Ks=Ks, d=d, times=10, save_path=save_path, multiprocessing=False, only_assortative=True, method=method)


if __name__ == "__main__":
    # main0()
    # main1()
    # main2()
    # main3()
    # main4()
    # main5()
    # main6()
    # main7()
    # main8()
    # main9()
    # main10()
    # main11()
    # main12()
    # main13()
    # main14()
    # main15()
    # main16()
    # main17()
    # main18()
    # main19()
    # main20()
    # main21()
    # main22()
    # main23()
    # main24()
    # main25()
    # main26()
    # main27()
    # main28()
    main29()
