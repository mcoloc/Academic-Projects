# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from agents import *
import gurobipy as grb
from scipy import spatial


class ExactVRPAgent(Agent):

    def __init__(self, env):
        self.env = env
        self.name = 'ExactAgent'
        self.quantile = 0.5
        self.data_improving_quantile = {
            "quantile": [],
            "performace": []
        }

    def compute_delivery_to_crowdship(self, deliveries):

        import json
        from sklearn.cluster import DBSCAN
        import random as rand   
        import math


        fp = open("./cfg/setting_1.json", 'r')
        settings = json.load(fp)
        fp.close()
        self.setting = settings

        if len(deliveries) == 0:
            return []
        
        points = []
        p_failure=[]
        p_time_windows={}
        
        self.delivery = []
        for _, ele in deliveries.items():
            points.append([ele['lat']-self.setting["depot"][0], ele['lng']-self.setting["depot"][1]])
            self.delivery.append(ele)
            p_failure.append(ele['p_failed'])
            p_time_windows[ele['id']]= [ele['time_window_min'], ele['time_window_max']] 

        distance_matrix = spatial.distance_matrix(points, points)
        print('DISTANCE\n',distance_matrix)

        print('CLUSTERING\n')
        clustering=DBSCAN(eps=self.EPS, min_samples=2).fit(points)
        print(clustering.labels_)
        #computing avg distances
        a,Npoints_per_cluster=np.unique(clustering.labels_,return_counts=True)
        max_distance=np.zeros(len(Npoints_per_cluster)-1)
        points_distance_from_depot=np.zeros(len(points))

        new_labels = clustering.labels_    

        clustering_tw = np.zeros((len(clustering.labels_)+1,3))

        for p in range(1,len(points)+1):
                clustering_tw[p][0] = clustering.labels_[p-1]
                clustering_tw[p][1] = p_time_windows[p][0] 
                clustering_tw[p][2] = p_time_windows[p][1] 

        clustering_counter = len(set(clustering.labels_)) - 1

        clu = clustering_counter
        clu_ = clustering_counter
        clu = 1


        new_pp = []


        for i in range(len(Npoints_per_cluster)-1): 
            #scorro tutti i cluster, saltando il primo perché indica gli outliers
            lab_list=clustering.labels_.tolist()
            pp = np.zeros((lab_list.count(i),4))


            count=0

            for p in range(len(points)):
                if(clustering.labels_[p]) == i:
                    pp[count][0] = p_time_windows[p+1][0] 
                    pp[count][1] = p_time_windows[p+1][1]
                    pp[count][2] = p  
                    count += 1

            pp=pp[pp[:,0].astype(float).argsort()]


            #clu = 1

            pp[0][3] = 0

            for j in range(count):
                flag = 0
                for k in range(j+1, count):
                    if pp[k][3] == 0 and pp[j][0] <= pp[k][0] and pp[j][1] >= pp[k][0]:
                        pp[k][3]=clu
                        pp[j][3]=clu
                    else:
                    #    flag = 1
                #if flag == 1:
                        clu += 1

            for pp_ in pp:
                new_pp.append(pp_)

        new_pp_array = np.zeros((len(points),1))

        for pp_ in new_pp:
            new_pp_array[int(pp_[2])] = int(pp_[3])

        print('questo:\n')
        print(new_pp_array)
        print('vecchio:\n')
        print(clustering.labels_)

        for i in range(len(points)):

            new_labels[i] = clustering.labels_[i]

            if new_labels[i] != -1:

                for j in range(i, len(points)):                
                    if new_pp_array[j] != 0 and clustering.labels_[i] == clustering.labels_[j] and new_pp_array[i] == new_pp_array[j]:
                        new_labels[j] = new_pp_array[i]
        
        distances = np.zeros((len(points),1))

        clusters = {}        

        for c in set(new_labels.tolist()):
            clusters[c]={'id':c, 'distance':0, 'points':[], 'lat':-1, 'lng':-1, 'failure':0}
            for p in range(len(points)):
                if new_labels[p] == c:
                    #clusters[c]['distance'] += math.sqrt(points[p][0]**2+points[p][1]**2)*(1-p_failure[p])
                    if abs(points[p][0])>clusters[c]['lat']:
                        clusters[c]['lat'] = abs(points[p][0])
                    if abs(points[p][1])>clusters[c]['lng']:
                        clusters[c]['lng'] = abs(points[p][1])
                    clusters[c]['failure'] += p_failure[p]
                    clusters[c]['points'].append(p)
        
        print('here')

        print(clusters, '\n\n')

        id_to_crowdship = []
        for c in set(new_labels.tolist()):
            clusters[c]['distance'] = math.sqrt(clusters[c]['lat']**2+clusters[c]['lng']**2)*(1-clusters[c]['failure']/len(clusters[c]['points']))
            if clusters[c]['id'] != -1 and clusters[c]['distance'] > 0.01:   #threshold to be improved
                for p_ in clusters[c]['points']:
                    id_to_crowdship.append(p_)

        print(clusters, '\n\n')
        print('crowd', sorted(id_to_crowdship))

        time.sleep(10)   

        return sorted(id_to_crowdship)

    



    def compute_VRP(self, delivery_to_do, vehicles_dict, gap=None, time_limit=None, verbose=False, debug_model=False):

        nodes = {}
        to_assign = []     
        to_assign_array = np.zeros((len(delivery_to_do.items()),2), int)

        c_= 0
        for key, ele in delivery_to_do.items():
            nodes[ele['id']] = {'id': ele['id'], 'tw_min': ele['time_window_min'], 'tw_max': ele['time_window_max'], 
                                'lat': ele['lat'], 'lng': ele['lng'], 'vol': ele['vol']}
            #to_assign.append(ele['id'])    
            to_assign_array[c_][0] = ele['time_window_min']
            to_assign_array[c_][1] = ele['id']
            c_+=1 

        to_assign_array=to_assign_array[to_assign_array[:,0].astype(float).argsort()]
        for c in range(len(delivery_to_do.items())):
            to_assign.append(to_assign_array[c][1])

        print(to_assign_array, to_assign)

        print('\n\n\nCOMPUTER VRP\n\n\n')     

        subsets = []

        print(to_assign)

        while 1:

            sub = {'tw_min': -1, 'tw_max': 999, 'sub': [], 'v':-1}
            sub['sub'].append(to_assign[0])
            print(sub['sub'],nodes[sub['sub'][0]]['tw_min'])
            to_assign.pop(0)
            for n in to_assign:
                ok = 1
                for node in sub['sub']:
                    #time.sleep(1)

                    #if ok == 1 and ((nodes[node]['tw_min'] < nodes[n]['tw_min'] and nodes[node]['tw_max'] > nodes[n]['tw_min']) or (nodes[node]['tw_min'] > nodes[n]['tw_min'] and nodes[node]['tw_min'] < nodes[n]['tw_max'])) :
                    #    ok = 1
                    if ok == 1 and ((nodes[node]['tw_min'] == nodes[n]['tw_min']) or (nodes[node]['tw_min'] == (nodes[n]['tw_min']+2)) or (nodes[node]['tw_min'] == (nodes[n]['tw_min']+1))) :
                        ok=1
                    else: 
                        ok = 0

                if ok == 1:
                    sub['sub'].append(n)
                    min_ = max(sub['tw_min'], nodes[node]['tw_min'])
                    max_ = min(sub['tw_max'], nodes[node]['tw_max'])
                    sub['tw_min']=min_
                    sub['tw_max']=max_
                    print(nodes[n], min_, max_)
                    #time.sleep(2)
                    to_assign.remove(n)

            if(len(sub['sub'])==1):
                sub['tw_min'] = nodes[sub['sub'][0]]['tw_min']
            print('\n\n')
            subsets.append(sub)
            if(len(to_assign)==0):
                break


        vehicles = np.zeros((len(vehicles_dict),4))
        v_index = 0
        for v in vehicles_dict:
            vehicles[v_index][0] = v['capacity']
            vehicles[v_index][1] = v['cost']
            vehicles[v_index][2] = 99999
            vehicles[v_index][3] = -1

            v_index += 1

        vehicles=vehicles[vehicles[:,0].astype(float).argsort()]
        print('vehicles=\n', len(vehicles))

        deliveries = {}
        c=0
        for sub in subsets:
            volume = 0
            for n in sub['sub']:
                volume += nodes[n]['vol']

            flag = 0

            for v in range(len(vehicles)):
                if vehicles[v][0] >= volume and (sub['tw_min'] > vehicles[v][3] or sub['tw_max'] < vehicles[v][2]):
                    flag = 1
                    break
            
            if flag == 1:
                deliveries[c]={'vehicles_capacity': vehicles[v][0], 'nodes': sub['sub'], 'v':v , 'tw_min': sub['tw_min'], 'tw_max':sub['tw_max'], 'id':c}
                c += 1

            else:
                subs = []
                while(len(sub)>0):
                    vol_ = 0
                    sub_sub = []
                    deliveries[c]['vehicles_capacity'] = vehicles[len(vehicles-1)-len(subs)]
                    while vol_ < vehicles[len(vehicles-1)-len(subs)] and len(sub)>0:
                        vol_ += nodes[sub[0]]['vol']
                        sub_sub.append(sub[0])
                        sub.pop(0)
                    subs.append(sub_sub)
                    deliveries[c]['nodes'] = sub_sub
                    c+=1

        del_order = np.zeros((len(deliveries),2))
        for deli in deliveries:
            print(deli,deliveries[deli])
            del_order[deli][0] = deliveries[deli]['tw_min']
            del_order[deli][1] = deliveries[deli]['id']
        
        del_order = del_order[del_order[:,0].astype(float).argsort()]

        import six
        import sys
        sys.modules['sklearn.externals.six'] = six
        import mlrose 

        reader = open("./data/distance_matrix.csv")
        x = np.genfromtxt(reader, delimiter=",")
        self.distance_matrix = x

        solution_final = []
        for i in range(len(vehicles)):
            solution_final.append([])

        t1 = time.time()


        for d in range(len(deliveries)):
            deli = del_order[d][1]
            coords_list=[]
            coords_list.append((self.setting["depot"][0],self.setting["depot"][1]))
            for node in deliveries[deli]['nodes']:
                coords_list.append((self.delivery_info[node]['lat'], self.delivery_info[node]['lng']))
            
            fitness_coords = mlrose.TravellingSales(coords = coords_list)

            dist_list = []
            for node in deliveries[deli]['nodes']:
                dist_list.append((0, node, self.distance_matrix[0][node]))
                for node_ in deliveries[deli]['nodes']:
                    dist_list.append((node, node_, self.distance_matrix[node][node_]))

            problem_fit = mlrose.TSPOpt(length=len(deliveries[deli]['nodes'])+1, fitness_fn = fitness_coords, maximize = False)

            """"
            best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)

            print(deliveries[deli]['nodes'], '\n',best_state)

            solution = []
            for p in best_state:
                print(p)
                if(p != 0):
                    solution.append(deliveries[deli]['nodes'][p-1])
                else:
                    solution.append(0)

            print('MLROSE', best_state, solution, best_fitness)

            """
            best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2, max_attempts = 10, random_state = 2)
            #best_state, best_fitness = mlrose.simulated_annealing(problem_fit, schedule=mlrose.ArithDecay(), max_attempts=1, max_iters=150, init_state=None, curve=False, random_state=5)
            print('MLROSE', best_state, best_fitness)

            solution = []
            for p in best_state:
                print(p)
                if(p != 0):
                    solution.append(deliveries[deli]['nodes'][p-1])
                else:
                    solution.append(0)

            solution = np.array(solution)

            depot_index = solution.tolist().index(0)
            sol = np.zeros(len(solution), int)
            
            for p in solution:
                if solution.tolist().index(p)>=depot_index:
                    sol[solution.tolist().index(p)-depot_index] = int(p)
                else:
                    sol[len(solution) + solution.tolist().index(p) - depot_index] = int(p)

            if len(solution_final[deliveries[deli]['v']]) > 0:
                solution_final[deliveries[deli]['v']] = solution_final[deliveries[deli]['v']] + (sol[1:len(sol)].tolist())
            else:
                solution_final[deliveries[deli]['v']] = solution_final[deliveries[deli]['v']] + (sol.tolist())


        for i in range(len(solution_final)):
            if (len(solution_final[i])) > 0:
                solution_final[i] = solution_final[i] + [0]
        print('\n\nSOLUTION=\n',solution_final)
        t = time.time() - t1
        print('TIME:', t)



        return solution_final






    def learn_and_save(self):
        #self.quantile = np.random.uniform()

        #id_deliveries_to_crowdship = self.compute_delivery_to_crowdship(self.env.get_delivery())
        #remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)
        #VRP_solution = self.compute_VRP(remaining_deliveries, self.env.get_vehicles())
        #obj = self.env.evaluate_VRP(VRP_solution)

        #self.data_improving_quantile['quantile'].append(self.quantile)
        #self.data_improving_quantile['performace'].append(tot_crowd_cost + obj)
            self.delivery_info = {}
            points = []
            import json
            file1 = open('./data/delivery_info.json')
            data = json.load(file1)
            for i in data.keys():
                if i == '0':
                    continue
                tmp = data[i]
                self.delivery_info[int(i)] = {
                    'id': int(tmp['id']),
                    'lat': float(tmp['lat']),
                    'lng': float(tmp['lng']),
                    'crowdsourced': 0,
                    'vol': float(tmp['vol']),
                    'crowd_cost': float(tmp['crowd_cost']),
                    'p_failed': float(tmp['p_failed']),
                    'time_window_min': float(tmp['time_window_min']),
                    'time_window_max': float(tmp['time_window_max']),
                }
                points.append([float(tmp['lat']), float(tmp['lng'])])
        
            self.EPS = self.computeEPS(self.delivery_info)
    
    def start_test(self):
        pos_min = self.data_improving_quantile['performace'].index(min(self.data_improving_quantile['performace']))
        self.quantile = self.data_improving_quantile['quantile'][pos_min]

    def computeEPS(self, deliveries):
        import json
        from sklearn.cluster import DBSCAN
        import random as rand   
        import math


        fp = open("./cfg/setting_1.json", 'r')
        settings = json.load(fp)
        fp.close()
        self.setting = settings

        if len(deliveries) == 0:
            return []
        
        points = []
        
        for _, ele in deliveries.items():
            points.append([ele['lat']-self.setting["depot"][0], ele['lng']-self.setting["depot"][0]])

        distance_matrix = spatial.distance_matrix(points, points)
        print('DISTANCE\n',distance_matrix)

        min_c=len(points)*0.2
        print('CLUSTERING\n')
        eps_=0.0001
        while True:
            clustering = DBSCAN(eps=eps_, min_samples=2).fit(points)
            if (list(clustering.labels_).count(-1) < min_c):
                break
            eps_ = eps_*2
        
        print(clustering.labels_)

        return eps_


        


