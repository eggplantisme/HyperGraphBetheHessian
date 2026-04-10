import itertools
import os
import numpy as np
from scipy.sparse import eye, diags, issparse, csr_array, find, hstack, vstack, csc_array, identity, kron
from net_data.enron.enron_parser import *
import itertools
import pickle
from _HyperCommunityDetection import *
import time
import json
import re
from tqdm import tqdm
import pandas


def cd(empiricalhg, save_path=None, visual_path=None, redetect=True, only_assortative=False, givenNumGroup=None,
       consider_ks=None, dc=False):
    if redetect or save_path is None:
        if dc is False:
            BH_Partition, BH_NumGroup = HyperCommunityDetect().BetheHessian(empiricalhg,
                                                                            num_groups=givenNumGroup,
                                                                            only_assortative=only_assortative,
                                                                            consider_ks=consider_ks)
            print(f'Bethe Hessian detect {BH_NumGroup} communities in network {empiricalhg.name}')
        else:
            BH_Partition, BH_NumGroup = HyperCommunityDetect().DCBetheHessian(empiricalhg, num_groups=givenNumGroup)
            print(f'Degree Corrected Bethe Hessian detect {BH_NumGroup} communities in network {empiricalhg.name}')
        if save_path is not None:
            with open(save_path, 'wb') as fw:
                pickle.dump(BH_Partition, fw)
    else:
        if save_path is not None:
            with open(save_path, 'rb') as fr:
                BH_Partition = pickle.load(fr)
                BH_NumGroup = np.size(np.unique(BH_Partition))
    if visual_path is not None:
        with open(visual_path, 'w') as fw:
            result = []
            for i in range(empiricalhg.n):
                result.append((i, BH_Partition[i], empiricalhg.meta[i]))
            result.sort(key=lambda term: term[1])
            for term in result:
                fw.write(f'{term[0]} {term[1]} {term[2]}\n')
    return BH_Partition, BH_NumGroup


class EmpiricalHyperGraph:
    def __init__(self, name, force=False):
        self.name = name
        self.H = None
        self.n = 0
        self.e = 0
        self.Ks = []
        self.meta = dict()
        self.construct(force)
        print(f'Construct {name} hypergraph with {self.n} nodes, {self.e} hyperedges and all possible k is {self.Ks}.')

    def construct(self, force=False):
        if self.name == 'enron':
            self.enron(force)
        elif self.name.startswith('yelp'):
            # "yelp_[city's name]" or just "yelp"
            self.yelp(force)
        elif self.name == 'tagMathSX':
            self.tagMathSX(force)
        elif self.name == 'tagAskUbuntu':
            self.tagAskUbuntu(force)
        elif self.name == 'ndc':
            self.ndc(force)
        elif self.name == 'primary':
            self.primary(force)
        elif self.name == 'highschool':
            self.highschool(force)
        elif self.name == 'highschool_cg':
            self.highschool(force, cg=True)
        elif self.name == 'coauthDBLP':
            self.coauthorDBLP(force)
        elif self.name == 'coauthAPS':
            self.coauthorAPS(force)
        else:
            print("Please check the name~")

    def enron(self, force=False):
        current_path = os.getcwd()
        os.chdir("./net_data/enron/")
        load_path = './enron_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                enron_data = pickle.load(fr)
                self.H = enron_data['H']
                self.n = enron_data['n']
                self.e = enron_data['e']
                self.Ks = enron_data['Ks']
                self.meta = enron_data['meta']
                # Get n, e, Ks
        else:
            start_parse = time.time()
            _, links = get_links(hyper=True)
            print(f'Parsing time {time.time() - start_parse}')
            ori_meta = dict()
            with open("./enron_employee.txt", 'r') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    temp = re.split(r'  +', line[1].strip())
                    ori_meta[line[0]] = temp[1:] if len(temp)>=2 else None
            address_id = dict()
            hyperedges = set()
            data = []
            row_ind = []
            col_ind = []
            for link in links:
                hyperedge = link[1]
                for address in hyperedge:
                    if address not in address_id.keys():
                        address_id[address] = self.n
                        email_name = address.split('@')[0]
                        if email_name not in ori_meta:
                            self.meta[self.n] = None
                            print(f'{email_name}', end=' ')
                        else:
                            self.meta[self.n] = ori_meta[email_name]
                        self.n += 1
                hyperedge = tuple(np.unique([address_id[address] for address in hyperedge]))  # no repeat nodes
                if len(hyperedge) <= 1:
                    continue  # skip hyperedge with order 1 (sender self send)
                repeat_exist = False
                for he in itertools.permutations(hyperedge):
                    if he in hyperedges:
                        repeat_exist = True  # no repeat hyperedges
                if repeat_exist:
                    continue
                else:
                    hyperedges.add(hyperedge)
                    k = np.size(hyperedge)
                    if k not in self.Ks:
                        self.Ks.append(k)
                    data += [1] * k
                    row_ind += list(hyperedge)
                    col_ind += [self.e] * k
                    self.e += 1
            with open('./addresses_id.txt', 'w') as fw:
                for a in address_id.keys():
                    fw.write(f'{a} {address_id[a]}\n')
            self.H = csr_array((data, (row_ind, col_ind)))
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)
        os.chdir(current_path)

    def yelp(self, force=False):
        load_path = './net_data/yelp/yelp_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                yelp_data = pickle.load(fr)
                self.H = yelp_data['H']
                self.n = yelp_data['n']
                self.e = yelp_data['e']
                self.Ks = yelp_data['Ks']
                self.meta = yelp_data['meta']
                # Get n, e, Ks
        else:
            # Load business (nodes)
            business_map = dict()
            meta_map = dict()
            with open("./net_data/yelp/yelp_dataset-2/yelp_academic_dataset_business.json", 'r',
                      encoding='utf-8') as fr:
                i = 0
                for line in tqdm(fr.readlines(), desc='Load Business'):
                    line = line.strip()
                    business_json = json.loads(line)
                    meta = dict()
                    if business_json['categories'] is not None:
                        meta['categories'] = [c.strip() for c in business_json['categories'].split(',')]
                    else:
                        meta['categories'] = []
                    meta['state'] = business_json['state']
                    meta['city'] = business_json['city']
                    meta['longitude'] = business_json['longitude']
                    meta['latitude'] = business_json['latitude']
                    business_id = business_json['business_id']
                    if business_id not in business_map.keys():
                        business_map[business_id] = {'id': i, 'meta': meta}
                        meta_map[i] = meta
                        i += 1
            print(f'Total number of nodes(bussiness) is {i}, '
                  f'with {len(np.unique([business_map[b]["meta"]["city"] for b in business_map]))} citys and '
                  f'{len(np.unique([business_map[b]["meta"]["state"] for b in business_map]))} states.')
            # print(f'city names:{[name for name in np.unique([business_map[b]["meta"]["city"] for b in business_map])]}')
            # Load review (hyperedges)
            data = []
            row_ind = []
            col_ind = []
            user_map = dict()
            with open("./net_data/yelp/yelp_dataset-2/yelp_academic_dataset_review.json", 'r',
                      encoding='utf-8') as fr:
                i = 0
                for line in tqdm(fr.readlines(), desc='Load Review'):
                    line = line.strip()
                    review_json = json.loads(line)
                    user_id = review_json['user_id']
                    business_id = review_json['business_id']
                    if user_id not in user_map.keys():
                        user_map[user_id] = {'num_review': 1, 'first_review_business_id': business_id}
                    else:
                        user_map[user_id]['num_review'] += 1
                        if 'id' not in user_map[user_id].keys():
                            user_map[user_id]['id'] = i   # only user who review at least 2 times will have an id
                            i += 1
                    if business_id not in business_map.keys():
                        print(f"Unexpected business_id {business_id} in review!")
                        break
                    if user_map[user_id]['num_review'] < 2:
                        pass  # only consider hyperedge with size >= 2
                    else:
                        if user_map[user_id]['num_review'] == 2:
                            data.append(1)
                            row_ind.append(business_map[user_map[user_id]['first_review_business_id']]['id'])
                            col_ind.append(user_map[user_id]['id'])
                        data.append(1)
                        row_ind.append(business_map[business_id]['id'])
                        col_ind.append(user_map[user_id]['id'])
            self.H = csr_array((data, (row_ind, col_ind)))
            # H = csc_array((data, (row_ind, col_ind)))
            # print(f'yelp before filter repeat hyper edge: {H.shape[0]} nodes, {H.shape[1]} hyperedges.')
            # self.H = H[:, [0]]
            # # Remove repeat hyperedge
            # for i in tqdm(range(1, H.shape[1]), desc="Check Hyperedge"):
            #     exist_same = False
            #     for j in range(0, self.H.shape[1]):
            #         if (H[:, [i]] - self.H[:, [j]]).sum() == 0:
            #             exist_same = True
            #             break
            #     if exist_same is False:
            #         self.H = hstack([self.H, H[:, [i]]])
            # self.H = csr_array(self.H)
            self.n = self.H.shape[0]
            self.e = self.H.shape[1]
            self.Ks = np.unique(self.H.sum(0)).tolist()
            self.meta = meta_map
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)
            # Save addition data
            with open('./net_data/yelp/bussiness_id.txt', 'w') as fw:
                for b in business_map.keys():
                    fw.write(f'{b} {business_map[b]["id"]} {business_map[b]["meta"]}\n')

    def tagMathSX(self, force=False):
        load_path = './net_data/tags-math-sx/tagMathSX_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                _data = pickle.load(fr)
                self.H = _data['H']
                self.n = _data['n']
                self.e = _data['e']
                self.Ks = _data['Ks']
                self.meta = _data['meta']
                # Get n, e, Ks
        else:
            nodes_map = dict()  # map from origin id to out id from 0
            # hyperedges = set()
            hyperedges_ori = set()
            data = []
            row_ind = []
            col_ind = []
            # node_label = dict()
            with open("./net_data/tags-math-sx/tags-math-sx-nverts.txt", 'r') as fr_nvert:
                with open("./net_data/tags-math-sx/tags-math-sx-simplices.txt", 'r') as fr_hedge:
                    ori_node_label = dict()
                    with open("./net_data/tags-math-sx/tags-math-sx-node-labels.txt", 'r') as fr_nl:
                        for line in fr_nl.readlines():
                            line = line.strip().split(' ')
                            ori_node_label[int(line[0])] = line[1]
                    for line in tqdm(fr_nvert.readlines(), desc='Load Hyperedges'):
                        # Load hyperedge
                        hedge_size = int(line.strip())
                        hedge = []
                        for i in range(hedge_size):
                            v = int(fr_hedge.readline().strip())
                            hedge.append(v)
                        hedge = tuple(hedge)
                        hedge_size = np.size(np.unique(hedge))  # No repeat nodes
                        # Filter Hyperedges
                        if hedge_size <= 1:  # Only consider hedge with order>1
                            continue
                        repeat_exist = False
                        for he in itertools.permutations(hedge):
                            if he in hyperedges_ori:
                                repeat_exist = True  # no repeat hyperedges
                        if repeat_exist is True:
                            continue
                        else:
                            hyperedges_ori.add(hedge)
                        # ReID nodes and update node_label
                        hedge = list(hedge)
                        for vi, v in enumerate(hedge):
                            if v not in nodes_map:
                                nodes_map[v] = self.n
                                self.meta[self.n] = ori_node_label[v]
                                self.n += 1
                            hedge[vi] = nodes_map[v]
                        # Add hyperedge
                        # hyperedges.add(hedge)
                        k = np.size(hedge)
                        if k not in self.Ks:
                            self.Ks.append(k)
                        data += [1] * k
                        row_ind += hedge
                        col_ind += [self.e] * k
                        self.e += 1
            # Construct H and save
            self.H = csr_array((data, (row_ind, col_ind)))
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)
            # Save addition data
            with open('./net_data/tags-math-sx/TagsId_map.txt', 'w') as fw:
                for b in nodes_map.keys():
                    fw.write(f'{b} {nodes_map[b]} {self.meta[nodes_map[b]]}\n')

    def tagAskUbuntu(self, force=False):
        load_path = './net_data/tags-ask-ubuntu/tagAskUbuntu_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                _data = pickle.load(fr)
                self.H = _data['H']
                self.n = _data['n']
                self.e = _data['e']
                self.Ks = _data['Ks']
                self.meta = _data['meta']
                # Get n, e, Ks
        else:
            nodes_map = dict()  # map from origin id to out id from 0
            # hyperedges = set()
            hyperedges_ori = set()
            data = []
            row_ind = []
            col_ind = []
            # node_label = dict()
            with open("./net_data/tags-ask-ubuntu/tags-ask-ubuntu-nverts.txt", 'r') as fr_nvert:
                with open("./net_data/tags-ask-ubuntu/tags-ask-ubuntu-simplices.txt", 'r') as fr_hedge:
                    ori_node_label = dict()
                    with open("./net_data/tags-ask-ubuntu/tags-ask-ubuntu-node-labels.txt", 'r') as fr_nl:
                        for line in fr_nl.readlines():
                            line = line.strip().split(' ')
                            ori_node_label[int(line[0])] = ' '.join(line[1:])
                    for line in tqdm(fr_nvert.readlines(), desc='Load Hyperedges'):
                        # Load hyperedge
                        hedge_size = int(line.strip())
                        hedge = []
                        for i in range(hedge_size):
                            v = int(fr_hedge.readline().strip())
                            hedge.append(v)
                        hedge = tuple(hedge)
                        hedge_size = np.size(np.unique(hedge))  # No repeat nodes
                        # Filter Hyperedges
                        if hedge_size <= 1:  # Only consider hedge with order>1
                            continue
                        repeat_exist = False
                        for he in itertools.permutations(hedge):
                            if he in hyperedges_ori:
                                repeat_exist = True  # no repeat hyperedges
                        if repeat_exist is True:
                            continue
                        else:
                            hyperedges_ori.add(hedge)
                        # ReID nodes and update node_label
                        hedge = list(hedge)
                        for vi, v in enumerate(hedge):
                            if v not in nodes_map:
                                nodes_map[v] = self.n
                                self.meta[self.n] = ori_node_label[v]
                                self.n += 1
                            hedge[vi] = nodes_map[v]
                        # Add hyperedge
                        # hyperedges.add(hedge)
                        k = np.size(hedge)
                        if k not in self.Ks:
                            self.Ks.append(k)
                        data += [1] * k
                        row_ind += hedge
                        col_ind += [self.e] * k
                        self.e += 1
            # Construct H and save
            self.H = csr_array((data, (row_ind, col_ind)))
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)
            # Save addition data
            with open('./net_data/tags-ask-ubuntu/TagsId_map.txt', 'w') as fw:
                for b in nodes_map.keys():
                    fw.write(f'{b} {nodes_map[b]} {self.meta[nodes_map[b]]}\n')

    def coauthorDBLP(self, force=False):
        load_path = './net_data/coauth-DBLP/coauthorDBLP_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                _data = pickle.load(fr)
                self.H = _data['H']
                self.n = _data['n']
                self.e = _data['e']
                self.Ks = _data['Ks']
                self.meta = _data['meta']
                # Get n, e, Ks
        else:
            nodes_map = dict()  # map from origin id to out id from 0
            # hyperedges = set()
            hyperedges_ori = set()
            data = []
            row_ind = []
            col_ind = []
            # node_label = dict()
            with open("./net_data/coauth-DBLP/coauth-DBLP-nverts.txt", 'r') as fr_nvert:
                with open("./net_data/coauth-DBLP/coauth-DBLP-simplices.txt", 'r') as fr_hedge:
                    ori_node_label = dict()
                    with open("./net_data/coauth-DBLP/coauth-DBLP-node-labels.txt", 'r') as fr_nl:
                        for line in fr_nl.readlines():
                            line = line.strip().split(' ')
                            ori_node_label[int(line[0])] = ' '.join(line[1:])
                    for line in tqdm(fr_nvert.readlines(), desc='Load Hyperedges'):
                        # Load hyperedge
                        hedge_size = int(line.strip())
                        hedge = []
                        for i in range(hedge_size):
                            v = int(fr_hedge.readline().strip())
                            hedge.append(v)
                        hedge = tuple(hedge)
                        hedge_size = np.size(np.unique(hedge))  # No repeat nodes
                        # Filter Hyperedges
                        if hedge_size <= 1:  # Only consider hedge with order>1
                            continue
                        # repeat_exist = False
                        # for he in itertools.permutations(hedge):
                        #     if he in hyperedges_ori:
                        #         repeat_exist = True  # no repeat hyperedges
                        # if repeat_exist is True:
                        #     continue
                        # else:
                        hyperedges_ori.add(hedge)
                        # ReID nodes and update node_label
                        hedge = list(hedge)
                        for vi, v in enumerate(hedge):
                            if v not in nodes_map:
                                nodes_map[v] = self.n
                                self.meta[self.n] = ori_node_label[v]
                                self.n += 1
                            hedge[vi] = nodes_map[v]
                        # Add hyperedge
                        # hyperedges.add(hedge)
                        k = np.size(hedge)
                        if k not in self.Ks:
                            self.Ks.append(k)
                        data += [1] * k
                        row_ind += hedge
                        col_ind += [self.e] * k
                        self.e += 1
            # Construct H and save
            self.H = csr_array((data, (row_ind, col_ind)))
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)
            # Save addition data
            with open('./net_data/coauth-DBLP/AuthorsId_map.txt', 'w') as fw:
                for b in nodes_map.keys():
                    fw.write(f'{b} {nodes_map[b]} {self.meta[nodes_map[b]]}\n')

    def coauthorAPS(self, force=False):
        load_path = './net_data/APS/coauthorAPS_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                _data = pickle.load(fr)
                self.H = _data['H']
                self.n = _data['n']
                self.e = _data['e']
                self.Ks = _data['Ks']
                self.meta = _data['meta']
                # Get n, e, Ks
        else:
            nodes_map = dict()  # map from origin id to our id from 0
            # hyperedges = set()
            data = []
            row_ind = []
            col_ind = []
            # node_label = dict()
            journals = []
            with open("./net_data/APS/aps/journals.csv", 'r') as fr_j:
                fr_j.readline()
                for line in tqdm(fr_j.readlines(), desc="Load Journals"):
                    journals.append((line.strip().split(',')[1], line.strip().split(',')[4],
                                     line.strip().split(',')[5]))  # (journalAbbr, issue, volume)
            publication_dois = []
            with open("./net_data/APS/aps/publications.csv", 'r') as fr_pub:
                fr_pub.readline()
                for line in tqdm(fr_pub.readlines(), desc="Load Publications"):
                    publication_dois.append((int(line.strip().split(',')[1]), line.strip().split(',')[3],
                                             line.strip().split(',')[3]))  # (journalId, doi, date)
            # Have a see the number of publication in different years
            NumPubinYear = dict()
            for i in range(len(publication_dois)):
                year = publication_dois[i][2].split('-')[0]
                if year in NumPubinYear:
                    NumPubinYear[year] += 1
                else:
                    NumPubinYear[year] = 0
            for year in NumPubinYear:
                print(f'{year} have {NumPubinYear[year]} publications')
            author_disamb = []
            with open("./net_data/APS/aps/author_names.csv", 'r', encoding="utf-8") as fr_author:
                fr_author.readline()
                for line in tqdm(fr_author.readlines(), desc='Load disambiguation authors'):
                    disamb_id = int(line.strip().split(',')[1])
                    name = line.strip().split(',')[2]
                    author_disamb.append((disamb_id, name))
            with open("./net_data/APS/aps/authorships.csv", 'r') as fr_hedge:
                fr_hedge.readline()  # skip first line
                temp_pid = -1
                temp_hedge = []
                for line in tqdm(fr_hedge.readlines(), desc='Load HyperEdge(authorships)'):
                    if temp_pid == -1:
                        temp_pid = int(line.strip().split(',')[1])
                        temp_hedge.append(int(line.strip().split(',')[2]))
                    elif int(line.strip().split(',')[1]) == temp_pid:
                        temp_hedge.append(int(line.strip().split(',')[2]))
                    else:
                        # Map from origin authoid to disambiguation id and remove repeat nodes
                        temp_hedge_after_disamb = np.unique([author_disamb[ori_nid][0]
                                                             for ori_nid in np.unique(temp_hedge)])
                        if np.size(temp_hedge_after_disamb) > 1:  # Filter publication with 1 author # TODO filter repeat hyperedge
                            hedge = temp_hedge_after_disamb.tolist()
                            for ori_nid in temp_hedge:
                                # Add node
                                disamb_id = author_disamb[ori_nid][0]
                                if disamb_id not in nodes_map.keys():
                                    nodes_map[disamb_id] = self.n
                                    self.n += 1
                                # if disamb_id not in hedge:
                                #     hedge.append(nodes_map[disamb_id])
                                # Add meta name
                                name = author_disamb[ori_nid][1]
                                if nodes_map[disamb_id] not in self.meta.keys():
                                    self.meta[nodes_map[disamb_id]] = dict({'name': [], 'affiliation': []})
                                if author_disamb[ori_nid][1] not in self.meta[nodes_map[disamb_id]]['name']:
                                    self.meta[nodes_map[disamb_id]]['name'].append(name)
                                # Add meta affiliation
                                doi = publication_dois[temp_pid][1]
                                journalId = publication_dois[temp_pid][0]
                                journalAbbr = journals[journalId][0]
                                issue = journals[journalId][1]
                                volume = journals[journalId][2]
                                if journalAbbr == 'PRXQUANTUM' and doi.split('.')[2] == '01':
                                    volume = "01"
                                with open(f'./net_data/APS/aps-dataset-metadata-2020/{journalAbbr}/{volume}/{doi.split("/")[1]}.json', 'r', encoding="utf-8") as fr_json:
                                    doi_data = json.load(fr_json)
                                    affiliationIds = []
                                    for a in doi_data['authors']:
                                        if a["name"] == name:
                                            if "affiliationIds" in a.keys():
                                                affiliationIds += a["affiliationIds"]
                                            else:
                                                pass
                                            break
                                    affiliations = []
                                    if 'affiliations' in doi_data:
                                        for i in affiliationIds:
                                            for a in doi_data['affiliations']:
                                                if a['id'] == i:
                                                    affiliations.append(a['name'])
                                                    break
                                    for a in affiliations:
                                        if a not in self.meta[nodes_map[disamb_id]]['affiliation']:
                                            self.meta[nodes_map[disamb_id]]['affiliation'].append(a)
                            # hyperedges.add(tuple(hedge))
                            k = np.size(hedge)
                            if k not in self.Ks:
                                self.Ks.append(k)
                            data += [1] * k
                            row_ind += hedge
                            col_ind += [self.e] * k
                            self.e += 1
                        temp_pid = int(line.strip().split(',')[1])
                        temp_hedge = [int(line.strip().split(',')[2])]
            # Construct H and save
            self.H = csr_array((data, (row_ind, col_ind)))
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)
            # Save addition data
            with open('./net_data/APS/AuthorsId_map.txt', 'w', encoding='utf-8') as fw:
                for b in nodes_map.keys():
                    fw.write(f'{b} {nodes_map[b]} {self.meta[nodes_map[b]]}\n')  # disambId, ourId, meta

    def ndc(self, force=False):
        load_path = './net_data/NDC-substances/ndc_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                _data = pickle.load(fr)
                self.H = _data['H']
                self.n = _data['n']
                self.e = _data['e']
                self.Ks = _data['Ks']
                self.meta = _data['meta']
                # Get n, e, Ks
        else:
            nodes_map = dict()  # map from origin id to out id from 0
            # hyperedges = set()
            hyperedges_ori = set()
            data = []
            row_ind = []
            col_ind = []
            # node_label = dict()
            with open("./net_data/NDC-substances/NDC-substances-nverts.txt", 'r') as fr_nvert:
                with open("./net_data/NDC-substances/NDC-substances-simplices.txt", 'r') as fr_hedge:
                    ori_node_label = dict()
                    with open("./net_data/NDC-substances/NDC-substances-node-labels.txt", 'r') as fr_nl:
                        for line in fr_nl.readlines():
                            line = line.strip().split(' ')
                            ori_node_label[int(line[0])] = " ".join(line[1:])
                    for line in tqdm(fr_nvert.readlines(), desc='Load Hyperedges'):
                        # Load hyperedge
                        hedge_size = int(line.strip())
                        hedge = []
                        for i in range(hedge_size):
                            v = int(fr_hedge.readline().strip())
                            hedge.append(v)
                        hedge = tuple(hedge)
                        hedge_size = np.size(np.unique(hedge))  # No repeat nodes
                        # Filter Hyperedges
                        if hedge_size <= 1:  # Only consider hedge with order>1
                            continue
                        repeat_exist = False
                        for he in itertools.permutations(hedge):  # TODO It take long time, if |hedge| is too large
                            if he in hyperedges_ori:
                                repeat_exist = True  # no repeat hyperedges
                        if repeat_exist is True:
                            continue
                        else:
                            hyperedges_ori.add(hedge)
                        # ReID nodes and update node_label
                        hedge = list(hedge)
                        for vi, v in enumerate(hedge):
                            if v not in nodes_map:
                                nodes_map[v] = self.n
                                self.meta[self.n] = ori_node_label[v]
                                self.n += 1
                            hedge[vi] = nodes_map[v]
                        # Add hyperedge
                        # hyperedges.add(hedge)
                        k = np.size(hedge)
                        if k not in self.Ks:
                            self.Ks.append(k)
                        data += [1] * k
                        row_ind += hedge
                        col_ind += [self.e] * k
                        self.e += 1
            # Construct H and save
            self.H = csr_array((data, (row_ind, col_ind)))
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)
            # Save addition data
            with open('./net_data/NDC-substances/SubstancesId_map.txt', 'w') as fw:
                for b in nodes_map.keys():
                    fw.write(f'{b} {nodes_map[b]} {self.meta[nodes_map[b]]}\n')

    def primary(self, force=False):
        load_path = './net_data/contact-primary-school/primary_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                _data = pickle.load(fr)
                self.H = _data['H']
                self.n = _data['n']
                self.e = _data['e']
                self.Ks = _data['Ks']
                self.meta = _data['meta']
                # Get n, e, Ks
        else:
            nodes_map = dict()  # map from origin id to out id from 0
            hyperedges = set()
            data = []
            row_ind = []
            col_ind = []
            with open("./net_data/contact-primary-school/hyperedges-contact-primary-school.txt", 'r') as fr_hedge:
                with open("./net_data/contact-primary-school/node-labels-contact-primary-school.txt", 'r') as fr_nl:
                    with open("./net_data/contact-primary-school/label-names-contact-primary-school.txt", 'r') as fr_ln:
                        label_name = dict()
                        for i, line in enumerate(fr_ln.readlines()):
                            line = line.strip()
                            label_name[i] = line
                        for i, line in enumerate(fr_nl.readlines()):
                            line = line.strip()
                            self.meta[i] = label_name[int(line)-1]
                for line in tqdm(fr_hedge.readlines(), desc='Load Hyperedges'):
                    # Load hyperedge
                    line = line.strip().split(',')
                    hedge = [int(i)-1 for i in line]
                    hedge = tuple(hedge)
                    hedge_size = np.size(np.unique(hedge))
                    # Filter Hyperedges
                    if hedge_size <= 1:  # Only consider hedge with order>1
                        continue
                    repeat_exist = False
                    for he in itertools.permutations(hedge):
                        if he in hyperedges:
                            repeat_exist = True  # no repeat hyperedges
                    if repeat_exist is True:
                        continue
                    else:
                        hyperedges.add(hedge)
                    # Count self.n
                    hedge = list(hedge)
                    for vi, v in enumerate(hedge):
                        if v not in nodes_map:
                            nodes_map[v] = self.n
                            self.n += 1
                    # Add hyperedge
                    k = hedge_size
                    if k not in self.Ks:
                        self.Ks.append(k)
                    data += [1] * k
                    row_ind += hedge
                    col_ind += [self.e] * k
                    self.e += 1
            # Construct H and save
            self.H = csr_array((data, (row_ind, col_ind)))
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)

    def highschool(self, force=False, cg=False):
        if cg is False:
            load_path = './net_data/contact-high-school/highschool_data.pkl'
        else:
            load_path = './net_data/contact-high-school-classgender/highschool_data.pkl'
        if os.path.exists(load_path) and force is False:
            with open(load_path, 'rb') as fr:
                _data = pickle.load(fr)
                self.H = _data['H']
                self.n = _data['n']
                self.e = _data['e']
                self.Ks = _data['Ks']
                self.meta = _data['meta']
                # Get n, e, Ks
        else:
            nodes_map = dict()  # map from origin id to out id from 0
            hyperedges = set()
            data = []
            row_ind = []
            col_ind = []
            hedge_f = "./net_data/contact-high-school-classgender/hyperedges-contact-high-school-classes-gender.txt" \
                if cg else "./net_data/contact-high-school/hyperedges-contact-high-school.txt"
            nl_f = "./net_data/contact-high-school-classgender/node-labels-contact-high-school-classes-gender.txt" \
                if cg else "./net_data/contact-high-school/node-labels-contact-high-school.txt"
            ln_f = "./net_data/contact-high-school-classgender/label-names-contact-high-school-classes-gender.txt" \
                if cg else "./net_data/contact-high-school/label-names-contact-high-school.txt"
            with open(hedge_f, 'r') as fr_hedge:
                with open(nl_f, 'r') as fr_nl:
                    with open(ln_f, 'r') as fr_ln:
                        label_name = dict()
                        for i, line in enumerate(fr_ln.readlines()):
                            line = line.strip()
                            label_name[i] = line
                        for i, line in enumerate(fr_nl.readlines()):
                            line = line.strip()
                            self.meta[i] = label_name[int(line)-1]
                for line in tqdm(fr_hedge.readlines(), desc='Load Hyperedges'):
                    # Load hyperedge
                    line = line.strip().split(',')
                    hedge = [int(i)-1 for i in line]
                    hedge = tuple(hedge)
                    hedge_size = np.size(np.unique(hedge))
                    # Filter Hyperedges
                    if hedge_size <= 1:  # Only consider hedge with order>1
                        continue
                    repeat_exist = False
                    for he in itertools.permutations(hedge):
                        if he in hyperedges:
                            repeat_exist = True  # no repeat hyperedges
                    if repeat_exist is True:
                        continue
                    else:
                        hyperedges.add(hedge)
                    # Count self.n
                    hedge = list(hedge)
                    for vi, v in enumerate(hedge):
                        if v not in nodes_map:
                            nodes_map[v] = self.n
                            self.n += 1
                    # Add hyperedge
                    k = hedge_size
                    if k not in self.Ks:
                        self.Ks.append(k)
                    data += [1] * k
                    row_ind += hedge
                    col_ind += [self.e] * k
                    self.e += 1
            # Construct H and save
            self.H = csr_array((data, (row_ind, col_ind)))
            with open(load_path, 'wb') as fw:
                pickle.dump(dict({'H': self.H, 'n': self.n, 'e': self.e, 'Ks': self.Ks, 'meta': self.meta}), fw)

    def get_operator(self, operator='BH', r=0, consider_ks=None):
        if operator == "BH":
            edge_order = self.H.sum(axis=0).flatten()
            D = None
            A = None
            if consider_ks is None:
                Ks = self.Ks
            else:
                Ks = consider_ks
            for k in tqdm(Ks, desc=rf'Construct $BH_{r}$'):
                edge_index = np.where(edge_order == k)[0]
                Hk = self.H[:, edge_index]
                Dk = diags(Hk.sum(axis=1).flatten().astype(float))
                Ak = Hk.dot(Hk.T) - diags(Hk.dot(Hk.T).diagonal())
                if D is None:
                    D = (k - 1) / ((1 - r) * (r + k - 1)) * Dk
                else:
                    D += (k - 1) / ((1 - r) * (r + k - 1)) * Dk
                if A is None:
                    A = r / ((1 - r) * (r + k - 1)) * Ak
                else:
                    A += r / ((1 - r) * (r + k - 1)) * Ak
            B = eye(D.shape[0]) - D + A
            return B
        elif operator == "NB":
            directed_hyperedge_size = self.H.sum()
            directed_hyperedges = []
            B = np.zeros((directed_hyperedge_size, directed_hyperedge_size))
            for mu in range(self.e):
                hedge = self.H[:, [mu]].nonzero()[0]
                for i in hedge:
                    directed_hyperedges.append((i, hedge))
            print(f'Non-backtrack constructing for {directed_hyperedge_size} directed node-hyperEdge pairs...')
            for index in tqdm(itertools.product(range(directed_hyperedge_size), repeat=2)):
                i = index[0]
                j = index[1]
                node_i = directed_hyperedges[i][0]
                node_j = directed_hyperedges[j][0]
                edge_i = directed_hyperedges[i][1]  # self.H[:, [directed_hyperedges[i][1]]].nonzero()[0]
                edge_j = directed_hyperedges[j][1]
                if node_j in edge_i and node_j != node_i and ((np.size(edge_j) != np.size(edge_i)) or ((edge_j == edge_i).all()) is np.False_):
                    B[i, j] = 1
                else:
                    B[i, j] = 0
            return csr_array(B)
        elif operator == "NB_":
            edge_order = self.H.sum(axis=0).flatten()
            D = None
            A = None
            Ks = self.Ks
            for k in tqdm(Ks, desc=rf'Construct $NB`$'):
                edge_index = np.where(edge_order == k)[0]
                Hk = self.H[:, edge_index]
                Dk = diags(Hk.sum(axis=1).flatten().astype(float))
                Ak = Hk.dot(Hk.T) - diags(Hk.dot(Hk.T).diagonal())
                if A is None:
                    A = csr_array(vstack([Ak for k in Ks]))
                else:
                    A = csr_array(hstack([A, csr_array(vstack([Ak for k in Ks]))]))
                if D is None:
                    D = csr_array(vstack([Dk for k in Ks]))
                else:
                    D = csr_array(hstack([D, csr_array(vstack([Dk for k in Ks]))]))
            I = csr_array(identity(len(Ks) * self.n))
            K = csr_array(diags(Ks))
            Zeros = csr_array(np.zeros((len(Ks) * self.n, len(Ks) * self.n)))
            righttop = D - csr_array(identity(len(Ks) * self.n))
            leftbottom = kron((csr_array(identity(len(Ks))) - K), csr_array(identity(self.n)), format='csr')
            rightbottom = A + kron((2 * csr_array(identity(len(Ks))) - K), csr_array(identity(self.n)), format='csr')
            B_ = csr_array(vstack([hstack([Zeros, righttop]), hstack([leftbottom, rightbottom])]))
            return B_

    def saveHedges(self, path):
        if self.H is not None and os.path.exists(path) is False:
            with open(path, 'w') as f:
                for e in range(self.H.shape[1]):
                    hedge = np.nonzero(self.H[:, [e]])[0]
                    for i in hedge:
                        f.write(f'{i} ')
                    f.write('\n')

    def saveDegrees(self, path):
        if self.H is not None and os.path.exists(path) is False:
            degree = [0] * self.n
            with open(path, 'w') as f:
                edge_order = self.H.sum(axis=0).flatten()
                for k in self.Ks:
                    edge_index = np.where(edge_order == k)[0]
                    Hk = self.H[:, edge_index]
                    Dk = Hk.sum(axis=1).flatten().astype(float)
                    for i in range(self.n):
                        if degree[i] == 0:
                            degree[i] = dict({k:Dk[i]})
                        else:
                            degree[i][k] = Dk[i]
                for i in range(self.n):
                    f.write(f'{i} ')
                    for k in degree[i].keys():
                        f.write(f'{k}:{degree[i][k]} ')
                    f.write('\n')


def main1():
    name = 'enron'
    # name = 'tagMathSX'
    # name = 'tagAskUbuntu'
    # name = 'ndc'
    # name = 'primary'
    # name = 'highschool'
    # name = 'highschool_cg'
    # name = 'yelp'
    # name = 'coauthDBLP'
    # name = 'coauthAPS'
    ehg = EmpiricalHyperGraph(name, force=False)
    givenNumGroup = 2
    only_assortative = True
    consider_ks = None
    visual_path = f'./result/hyperEmpirical/{name}_viewOfPartition' \
                  f'{f"_given{givenNumGroup}Groups" if givenNumGroup is not None else ""}' \
                  f'{f"_assort" if only_assortative else ""}' \
                  f'{f"{consider_ks}" if consider_ks is not None else ""}.txt'
    save_path = f'./result/hyperEmpirical/{name}_BHPartition' \
                f'{f"_given{givenNumGroup}Groups" if givenNumGroup is not None else ""}' \
                f'{f"_assort" if only_assortative else ""}' \
                f'{f"{consider_ks}" if consider_ks is not None else ""}.pkl'
    partition, _ = cd(ehg, save_path=save_path, visual_path=visual_path,
                      givenNumGroup=givenNumGroup, only_assortative=only_assortative, consider_ks=consider_ks)


def main2():
    # name = 'tagAskUbuntu'
    # name = "highschool"
    name = "enron"
    ehg = EmpiricalHyperGraph(name, force=False)
    # ehg.saveHedges(path=f'./net_data/contact-high-school/{name}_hyperedge.txt')
    ehg.saveDegrees(path=f'./net_data/contact-high-school/{name}_degree.txt')


def main3():
    name = 'highschool'
    ehg = EmpiricalHyperGraph(name, force=False)
    givenNumGroup = 9
    only_assortative = True
    consider_ks = None
    dc = True
    visual_path = f'./result/hyperEmpirical/{name}_viewOfPartition' \
                  f'{f"_given{givenNumGroup}Groups" if givenNumGroup is not None else ""}' \
                  f'{f"_assort" if only_assortative else ""}' \
                  f'{f"{consider_ks}" if consider_ks is not None else ""}' \
                  f'{"_dc" if dc else ""}.txt'
    save_path = f'./result/hyperEmpirical/{name}_BHPartition' \
                f'{f"_given{givenNumGroup}Groups" if givenNumGroup is not None else ""}' \
                f'{f"_assort" if only_assortative else ""}' \
                f'{f"{consider_ks}" if consider_ks is not None else ""}' \
                f'{"_dc" if dc else ""}.pkl'
    partition, _ = cd(ehg, save_path=save_path, visual_path=visual_path,
                      givenNumGroup=givenNumGroup, only_assortative=only_assortative, consider_ks=consider_ks,
                      dc=dc)

def debug():
    name = 'highschool'
    ehg = EmpiricalHyperGraph(name, force=False)
    givenNumGroup = 9
    sign = False
    # NB_ = ehg.get_operator('NB_')
    # print("get NB")
    NB_Partition, NB_NumGroup = HyperCommunityDetect().NonBackTracking_(ehg, num_groups=givenNumGroup, sign=sign)
    print(f'Nonbacktracking SC detect {NB_NumGroup} communities in network {name}')
    save_path = f'./result/hyperEmpirical/{name}_NBPartition' \
                f'{f"_given{givenNumGroup}Groups" if givenNumGroup is not None else ""}' \
                f'{f"_sign" if sign else "_noSign"}.pkl'
    with open(save_path, 'wb') as fw:
        pickle.dump(NB_Partition, fw)
    meta = []
    with open('./net_data/contact-high-school/highschool_data.pkl', 'rb') as fr:
        _data = pickle.load(fr)
        for i in range(_data['n']):
            meta.append(_data['meta'][i])
    meta = np.array(meta)
    meta_num = np.size(np.unique(meta))
    cm = np.zeros((NB_NumGroup, meta_num))
    uniquePartition = np.unique(NB_Partition)
    uniqueMeta = np.unique(meta)
    for iP in uniquePartition:
        trueIndex = np.where(NB_Partition == iP)[0]
        #         print(type(trueIndex))
        for iM in uniqueMeta:
            i = np.where(uniquePartition == iP)
            j = np.where(uniqueMeta == iM)
            cm[i, j] = np.size(np.where(meta[trueIndex] == iM))
    df = pandas.DataFrame(cm, uniquePartition, uniqueMeta)
    path = f'./result/hyperEmpirical/{name}_NBSC_cm' \
           f'_given{givenNumGroup}Groups_{"noSign" if sign is False else ""}.xlsx'
    with pandas.ExcelWriter(path) as writer:
        df.to_excel(excel_writer=writer)
        writer._save()
    pass


if __name__ == '__main__':
    # main1()
    # main2()
    main3()
    # debug()
