from math import exp
import time
import os
import numpy as np
from scipy.special import comb
from sklearn.metrics.cluster import adjusted_mutual_info_score
from _HyperSBM import *
from _HyperCommunityDetection import *
from _FigureJiazeHelper import get_confusionmatrix
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import Pool


def CDwithBH(hsbm, bipartite=False, projection=False, givenNumGroup=None):
    start = time.time()
    if projection:
        A = hsbm.H.dot(hsbm.H.T) - diags(hsbm.H.dot(hsbm.H.T).diagonal())
        bulk = np.sqrt((A**2).sum() / A.shape[0])
        BH_Partition, BH_NumGroup = HyperCommunityDetect().BetheHessian(num_groups=givenNumGroup, hsbm=hsbm, projectionMatrix=A, givenBulk=bulk)
    elif bipartite:
        # TODO Still onemode projection of bippartified hyper graph
        BH_Partition, BH_NumGroup = HyperCommunityDetect().BipartiteBH(hsbm, num_groups=hsbm.q)
    else:
        BH_Partition, BH_NumGroup = HyperCommunityDetect().BetheHessian(hsbm, num_groups=givenNumGroup)
    cd_time = time.time() - start
    cm, _ = get_confusionmatrix(hsbm.groupId, BH_Partition, hsbm.q, BH_NumGroup)
    ami = adjusted_mutual_info_score(hsbm.groupId, BH_Partition)
    print(f"BH result AMI: {ami}. Time={cd_time}. Confusion Matrix({np.shape(cm)}) is: \n{cm}")
    return ami, BH_NumGroup, cd_time


def CDwithBP(hsbm, arg=None, hypergraph_save_name=None):
    start = time.time()
    if arg is None:
        arg = dict()
        arg["q"] = hsbm.q
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
    BP_Partition, BP_NumGroup = HyperCommunityDetect().BeliefPropagation(hsbm, arg)
    cd_time = time.time() - start
    cm, _ = get_confusionmatrix(hsbm.groupId, BP_Partition, hsbm.q, BP_NumGroup)
    ami = adjusted_mutual_info_score(hsbm.groupId, BP_Partition)
    print(f"BP result AMI: {ami}. Time={cd_time}. Confusion Matrix({np.shape(cm)}) is: \n{cm}")
    return ami, BP_NumGroup, cd_time


def CDwithLouvain(hsbm):
    start = time.time()
    Partition, NumGroup = HyperCommunityDetect().modularity(hsbm.H)
    cd_time = time.time() - start
    # cm, _ = get_confusionmatrix(hsbm.groupId, Partition, hsbm.q, NumGroup)
    ami = adjusted_mutual_info_score(hsbm.groupId, Partition)
    print(f"Louvain result AMI: {ami}. Time={cd_time}. Number of groups: {NumGroup}")
    return ami, NumGroup, cd_time


def CDwithSCLaplacian(hsbm, num_groups):
    start = time.time()
    Partition, NumGroup = HyperCommunityDetect().SCLaplacian(hsbm, num_groups)
    cd_time = time.time() - start
    ami = adjusted_mutual_info_score(hsbm.groupId, Partition)
    print(f"SCLaplacian result AMI: {ami}. Time={cd_time}. Number of groups: {NumGroup}")
    return ami, NumGroup, cd_time

def exp_subprocess(n=3000, q=3, d=15, Ks=(2, ), epsilon=1, times=1, save_path=None, bipartite=False, projection=False,
                   bp=False, givenNumGroup=None, louvain=False, scLaplacian=False):
    sizes = [int(n / q)] * q
    ps_dict = dict()
    temp = 0
    for k in Ks:
        temp += q * comb(int(n/q), k) * k / (n**k) + epsilon * (comb(n, k) - q * comb(int(n/q), k)) * k / (n**k)
    cin = d / temp
    cout = epsilon * cin
    results = ""
    for t in range(times):
        start = time.time()
        if len(Ks) > 1:
            hsbm = UnUniformSymmetricHSBM(n, q, Ks, cin, cout)
        elif len(Ks) == 1:
            hsbm = UniformSymmetricHSBM(n, q, Ks[0], cin, cout)
        print(f'epsilon={epsilon} times={t} start. cin={cin}, cout={cout}, hsbm construct time={time.time()-start}')
        # Community Detection
        if bp:
            hypergraph_save_name = f'amiexp_n={n}_q={q}_d={d}_Ks={Ks}_epsilon={epsilon}_times={t}'
            result = CDwithBP(hsbm, hypergraph_save_name=hypergraph_save_name)
        elif louvain:
            result = CDwithLouvain(hsbm)
        elif scLaplacian:
            num_groups = hsbm.q  # Laplacian need number of groups given
            result = CDwithSCLaplacian(hsbm, num_groups)
        else:
            result = CDwithBH(hsbm, bipartite, projection, givenNumGroup=givenNumGroup)
        results += f'{epsilon} {t} {result[0]} {result[1]} {result[2]}\n'
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


def run_exp(epsilons, times, save_path=None, n=3000, q=3, d=15, Ks=(2, ), multiprocessing=True, bipartite=False,
            projection=False, bp=False, givenNumGroup=None, louvain=False, scLaplacian=False):
    epsilon_done = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                epsilon_done.add(round(float(row[0]), 5))
    if multiprocessing:
        p = Pool(2)
        for epsilon in epsilons:
            if round(epsilon, 5) in epsilon_done:
                print(f'snr={epsilon} has been run!')
                continue
            p.apply_async(exp_subprocess, args=(n, q, d, Ks, epsilon, times, save_path, bipartite, projection, bp, givenNumGroup, louvain, scLaplacian),
                          callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for epsilon in epsilons:
            if round(epsilon, 5) in epsilon_done:
                print(f'snr={epsilon} has been run!')
                continue
            savepath, results = exp_subprocess(n, q, d, Ks, epsilon, times, save_path, bipartite, projection, bp, givenNumGroup, louvain, scLaplacian)
            write_results((savepath, results))


def read_exp(load_path, add_paths=None, num_result=2):
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
        mean_Results = []
        std_Results = []
        for i in range(num_result):
            mean_Results.append(np.zeros(np.size(epsilons)))
            std_Results.append(np.zeros(np.size(epsilons)))
        i = 0
        for epsilon in epsilons:
            ami_results = results[np.squeeze(np.argwhere(results[:, 0] == epsilon))]
            if np.size(ami_results) == 0:
                print(f"Some parameter epsilon={epsilon} didn't run!")
            mean_ami = np.mean(ami_results, 0)[2:]
            std_ami = np.std(ami_results, 0)[2:]
            for nr in range(num_result):
                mean_Results[nr][i] = mean_ami[nr]
                std_Results[nr][i] = std_ami[nr]
            i += 1
    return epsilons, mean_Results, std_Results


def exp0():
    n = 100
    q = 2
    d = 15
    times = 5
    epsilons = np.concatenate((np.linspace(0.1, 1, 10), np.linspace(2, 10, 9)), axis=None)
    Ks = (2, 3)
    multiprocessing = False
    fileId = 'amiExpHyper24.5.23' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp1():
    n = 100
    q = 2
    d = 15
    times = 5
    epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    Ks = (3, )
    multiprocessing = True
    addtionTag = "_uniform"
    fileId = 'amiExpHyper24.5.24' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp2():
    n = 100
    q = 2
    d = 15
    times = 40
    epsilons = np.linspace(0.4, 0.8, 40)
    Ks = (2, 3)
    multiprocessing = True
    addStrId = f'_40more0.4~0.8'
    fileId = 'amiExpHyper24.5.30' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH' + addStrId
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp3():
    n = 100
    q = 2
    d = 15
    times = 40
    epsilons = np.linspace(0.1, 1, 20)
    Ks = (2, 3)
    bipartite = True
    multiprocessing = True
    addStrId = f'_0.4~0.8_40more'
    fileId = 'amiExpHyper24.5.31' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{"_bi" if bipartite else ""}' + addStrId
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bipartite)


def exp4():
    """ Exp q>2 """
    n = 150
    q = 3
    d = 15
    times = 40
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    bipartite = False
    multiprocessing = True
    # addStrId = f'_higher_q'
    addStrId = f'_40more'
    # fileId = 'amiExpHyper24.6.5' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{"_bi" if bipartite else ""}' + addStrId
    fileId = 'amiExpHyper24.6.30' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{"_bi" if bipartite else ""}' + addStrId
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bipartite)


def exp5():
    n = 100
    q = 2
    d = 10
    times = 40
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (3, )
    multiprocessing = True
    # addtionTag = ""
    addtionTag = "_40more"
    fileId = 'amiExpHyper24.6.29' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp6():
    n = 100
    q = 2
    d = 10
    times = 50
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    multiprocessing = True
    addtionTag = ""
    fileId = 'amiExpHyper24.6.30' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp7():
    """ Exp q>2 """
    n = 150
    q = 3
    d = 15
    times = 10
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    bipartite = False
    projection = True
    multiprocessing = True
    # addStrId = f'_higher_q'
    addStrId = f''
    # fileId = 'amiExpHyper24.6.5' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{"_bi" if bipartite else ""}' + addStrId
    fileId = 'amiExpHyper24.8.11' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{"_bi" if bipartite else ""}' \
                                    f'{"_proj" if projection else ""}' + addStrId
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bipartite, projection)


def exp8():
    n = 100
    q = 2
    d = 10
    times = 2
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    multiprocessing = True
    bp = True
    addtionTag = ""
    fileId = 'amiExpHyper24.12.09' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp)


def exp19():
    n = 30000
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 1, 101)
    Ks = (3, )
    multiprocessing = False
    bp = False
    addtionTag = ""
    fileId = 'amiExpHyper24.12.20' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp)


def exp20():
    n = 30000
    q = 2
    d = 10
    times = 20
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (3, )
    multiprocessing = True
    bp = False
    addtionTag = ""
    fileId = 'amiExpHyper24.12.20' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp)


def exp21():
    n = 30000
    q = 2
    d = 10
    times = 20
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.4, 0.5, 51)
    Ks = (3, )
    multiprocessing = True
    bp = False
    addtionTag = "0.4-0.5"
    fileId = 'amiExpHyper24.12.21' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp)


def exp22():
    n = 30000
    q = 2
    d = 10
    times = 30
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.4, 0.5, 51)
    Ks = (3, )
    multiprocessing = True
    bp = False
    addtionTag = "0.4-0.5more30"
    fileId = 'amiExpHyper24.12.21' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp)


def exp23():
    n = 30000
    q = 2
    d = 10
    times = 30
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (3, )
    multiprocessing = True
    bp = False
    addtionTag = "more30"
    fileId = 'amiExpHyper24.12.21' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp)


def exp24():
    n = 30000
    q = 2
    d = 10
    times = 30
    epsilons = np.linspace(0.4, 0.6, 101)
    # epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    multiprocessing = True
    addtionTag = "0.4~0.6more30"
    fileId = 'amiExpHyper24.12.22' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp25():
    n = 30000
    q = 3
    d = 10
    times = 40
    epsilons = np.linspace(0.3, 0.5, 101)
    # epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    multiprocessing = True
    addtionTag = "0.3~0.5more40"
    fileId = 'amiExpHyper24.12.23' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp26():
    n = 30000
    q = 3
    d = 10
    times = 50
    epsilons = np.linspace(0.3, 0.5, 101)
    # epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    multiprocessing = True
    addtionTag = "0.3~0.5more50"
    fileId = 'amiExpHyper24.12.24' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp27():
    n = 30000
    q = 2
    d = 10
    times = 50
    epsilons = np.linspace(0.4, 0.6, 101)
    # epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    multiprocessing = True
    addtionTag = "0.4~0.6more50"
    fileId = 'amiExpHyper24.12.24' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp28():
    n = 3000
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 0.9, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    projection = True
    addtionTag = "WBH_projection"
    fileId = 'amiExpHyper26.02.08' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing=multiprocessing, projection=projection, bp=bp)


def exp29():
    n = 3000
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 0.9, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    projection = True
    givenNumGroup = q
    addtionTag = "WBH_projection_givenNumGroup"
    fileId = 'amiExpHyper26.02.07' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing=multiprocessing, projection=projection, bp=bp, givenNumGroup=givenNumGroup)


def exp30():
    n = 100
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 0.9, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    projection = True
    addtionTag = "WBH_projection"
    fileId = 'amiExpHyper26.02.08' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_{"BP" if bp else "BH"}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing=multiprocessing, projection=projection, bp=bp)


def exp31():
    n = 3000
    q = 2
    d = 10
    times = 5
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 0.9, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    louvain = True
    addtionTag = ""
    fileId = 'amiExpHyper26.03.16' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_Louvain_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain)

def exp32():
    n = 3000
    q = 2
    d = 10
    times = 5
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 0.9, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    louvain = False
    addtionTag = ""
    fileId = 'amiExpHyper26.03.18' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_BH_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain)

def exp33():
    n = 3000
    q = 2
    d = 10
    times = 5
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 0.9, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = True
    louvain = False
    addtionTag = ""
    fileId = 'amiExpHyper26.03.18' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_BP_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain)

def exp34():
    n = 3000
    q = 2
    d = 10
    times = 5
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 0.9, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    louvain = False
    scLaplacian = True
    addtionTag = ""
    fileId = 'amiExpHyper26.03.20' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_SCLaplacian_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain, scLaplacian=scLaplacian)

# 0.4-0.6
def exp35():
    n = 3000
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.4, 0.6, 101)
    Ks = (2, 3)
    multiprocessing = True
    bp = False
    louvain = True
    addtionTag = "0.4~0.6more10"
    fileId = 'amiExpHyper26.03.20' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_Louvain_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain)

def exp36():
    n = 3000
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.4, 0.6, 101)
    Ks = (2, 3)
    multiprocessing = True
    bp = False
    louvain = False
    scLaplacian = True
    addtionTag = "0.4~0.6more10"
    fileId = 'amiExpHyper26.03.21' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_SCLaplacian_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain, scLaplacian=scLaplacian)

def exp37():
    n = 3000
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.4, 0.6, 101)
    Ks = (2, 3)
    multiprocessing = True
    bp = False
    louvain = False
    scLaplacian = False
    addtionTag = "0.4~0.6more10"
    fileId = 'amiExpHyper26.03.21' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_BH_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain, scLaplacian=scLaplacian)

def exp38():
    n = 3000
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.4, 0.6, 101)
    Ks = (2, 3)
    multiprocessing = True
    bp = True
    louvain = False
    scLaplacian = False
    addtionTag = "0.4~0.6more10"
    fileId = 'amiExpHyper26.03.21' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_BP_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain, scLaplacian=scLaplacian)

def exp39():
    n = 3000
    q = 2
    d = 10
    times = 5
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 0.9, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    louvain = False
    givenNumGroup = q
    addtionTag = ""
    fileId = 'amiExpHyper26.03.24' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_BH_givenNumGroup={givenNumGroup}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain, givenNumGroup=givenNumGroup)


def exp40():
    n = 30000
    q = 2
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.4, 0.6, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    louvain = True
    addtionTag = ""
    fileId = 'amiExpHyper26.03.26' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_Louvain_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain)


def exp41():
    n = 30000
    q = 3
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.3, 0.5, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    louvain = False
    scLaplacian = True
    addtionTag = ""
    fileId = 'amiExpHyper26.04.01' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_SCLaplacian_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bp=bp, louvain=louvain, scLaplacian=scLaplacian)


def debug():
    n = 30000
    q = 3
    d = 10
    times = 10
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.3, 0.5, 101)
    Ks = (2, 3)
    multiprocessing = False
    bp = False
    louvain = False
    scLaplacian = True
    addtionTag = ""
    fileId = 'amiExpHyper26.04.01' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}_SCLaplacian_{addtionTag}'
    load_path = "./result/detectabilityHyper/" + fileId + ".txt"
    sclap_epsilons, sclap_results = read_exp(load_path=load_path, add_paths=add_paths)


if __name__ == '__main__':
    # exp0()
    # debug()
    # exp1()
    # exp2()
    # exp3()
    # exp4()
    # exp5()
    # exp6()
    # exp7()
    # exp8()
    # exp19()
    # exp20()
    # exp21()
    # exp22()
    # exp23()
    # exp24()
    # exp25()
    # exp26()
    # exp27()
    # exp28()
    # exp29()
    # exp30()
    # exp31()
    # exp32()
    # exp33()
    # exp34()
    # exp35()
    # exp36()
    # exp37()
    # exp38()
    # exp39()
    # exp40()
    exp41()
