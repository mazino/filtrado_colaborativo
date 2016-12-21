#!usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import log
from sklearn.metrics.pairwise import cosine_similarity

# _Pearsonr: x e y tienen que ser del tipo np.array()
def _pearsonr(x, y):
    #productos que ha calificado x 
    x_index = np.nonzero(x)[0]
    y_index = np.nonzero(y)[0]

    mx = x[x_index].mean()
    my = y[y_index].mean()
  
    # Interseccion de los productos vistos por x e y
    interseccion = np.intersect1d(x_index,y_index)

    xm = x[interseccion] - mx
    ym = y[interseccion] - my
    
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(np.sum(xm*xm) * np.sum(ym*ym))

    if(r_den != 0):
        r = r_num / r_den
    else:
        r = 0
    return r

def hash_productos(productos, id_productos):
    i = 0
    hash_productos = {}
    for v in productos[id_productos]:
        hash_productos[i] = v-1
        i+=1
    return hash_productos

def _similitud(u_a, v, u_a_ratings, m_ratings, user_test, u_a_similarity, type='coseno'):
    if(type == 'coseno'):
        user_similarity = []
        # Usuarios que han calificado el producto v
        for u in m_ratings[:,v].nonzero()[0]:
            # Se filtran lo usuarios que pertenecen al conjunto de test
            if u not in user_test:
                if(u_a_similarity[u] == -2):
                    if(u != u_a):
                        temp = m_ratings[u].toarray()[0]
                        cosine = cosine_similarity([u_a_ratings],  [temp])[0][0]
                        user_similarity.append((u, cosine))
                        u_a_similarity[u] = cosine
                    else:
                        user_similarity.append((u,0))
                else:
                    user_similarity.append((u, u_a_similarity[u]))
            
    elif(type == 'pearson'):
        user_similarity = []
        # Usuarios que han visto el producto v
        for u in m_ratings[:,v].nonzero()[0]:
            # Se filtran lo usuarios que pertenecen al conjunto de test
            if u not in user_test:
                if(u_a_similarity[u] == -2):
                    if(u != u_a):
                        pearson = _pearsonr(u_a_ratings,  m_ratings[u].toarray()[0])
                        user_similarity.append((u, pearson))
                        u_a_similarity[u] = pearson
                    else:
                        user_similarity.append((u,0))
                else:
                    user_similarity.append((u, u_a_similarity[u]))
            
    return user_similarity

def user_mean_std(m_ratings, n_users):
    u_mean = np.zeros(n_users)
    u_std = np.zeros(n_users)
    for u in range(0, n_users):
        #aux = matrix[u].sum()/matrix[u].nnz # Promedio de ratings del usuario u 
        aux = m_ratings[u][m_ratings[u].nonzero()] # Array de los ratings del usuario u
        u_mean[u] = np.mean(aux)
        u_std[u] = np.std(aux)
    return u_mean, u_std

def one_user_mean_std(u_ratings):
    u_mean = np.mean(u_ratings)
    u_std = np.std(u_ratings)
    return u_mean, u_std

def k_neighborhood(user_similarity, k):
    # lista de usuarios odenador descendentemente de acuerdo a la similitud con el u_a
    user_similarity.sort(key=lambda tup: tup[1], reverse=True)

    u_neighbor = []
    i = 0
    for u in user_similarity:
        u_neighbor.append(u)
        i += 1
        if(i == k):
            break

    return u_neighbor

def _precision_recall(prediccion, relevantes, N, p_at_k, r_at_k):
    # se ordenan las predicciones en orden decreciente, segun valor de rating predecido.
    prediccion.sort(key=lambda tup: tup[1], reverse=True)
    pre_at_k = []
    rec_at_k = []
    # Se calcula los verdadero positivos
    true_positive = 0.0
    for k in range (0, N):
        if(prediccion[k][0] in relevantes):
            true_positive += 1
        pre_at_k.append(true_positive/(k+1))
        rec_at_k.append(true_positive/len(relevantes))
    p_at_k.append(pre_at_k)
    r_at_k.append(rec_at_k)

#nDCG
def _nDCG(user, prediccion, m_ratings, N, nDCG_fold):
    prediccion.sort(key=lambda tup: tup[1], reverse=True)
    # Obtenemos los ratings ordenados de manera decreciente del usuario
    ratings_ideal = []
    for i, (v, r) in enumerate(prediccion):
        ratings_ideal.append(m_ratings[user, v])
        
    ratings_ideal = np.sort(ratings_ideal)
    ratings_ideal = ratings_ideal[::-1]
    DCG = []
    DCG_ideal = []    
    DCG.append(prediccion[0][1])
    DCG_ideal.append(ratings_ideal[0])
    #print "prediccion = {:.4f} y rating real = {:}".format(prediccion[0][1], ratings_ideal[0])
    for k in range(1,N):
        DCG.append(prediccion[k][1]/log(k + 1, 2) + DCG[-1])
        DCG_ideal.append(ratings_ideal[k]/log(k + 1, 2) + DCG_ideal[-1])
        nDCG = np.divide(DCG, DCG_ideal)
        #print "prediccion = {:.4f} y rating real = {:}".format(prediccion[k][1],  ratings_ideal[k])
    nDCG_fold.append(nDCG)

# Crea 5 folds de la forma [test] [train], utilizando una distribucion uniforme
# cada lista contiene el indice de los usarios de test y de train
def k_fold_(matrix, n_users):
    #Se obtienen un arreglo de los usuarios ordenados por cantidad de ratings: users = (N° ratings, indice_usuario)
    users = []
    for u in range (0, n_users):
        users.append((matrix[u].nnz, u))
    users.sort()
    users = users[::-1]

    fold_1 = []
    fold_2 = []
    fold_3 = []
    fold_4 = []
    fold_5 = []
    
    k = 5
    # se construyen los fold de manera uniforme.
    for i, (n_ratings, u) in enumerate(users):
        if(i%k == 0):
            fold_1.append(u)
        elif(i%k == 1):
            fold_2.append(u)
        elif(i%k == 2):
            fold_3.append(u)
        elif(i%k == 3):
            fold_4.append(u)
        elif(i%k == 4):
            fold_5.append(u)
    # se crea una lista de los folds para facilitar el acceso a estos mediante un for
    folds = [fold_1, fold_2, fold_3, fold_4, fold_5]

    # Se crean una lista de la forma [test, train] usando cada fold como test al menos una vez
    k_folds = []

    for k, fold in enumerate(folds):
        train = []
        test = fold
        if(k == 0):
            for i in range(0, len(folds)):
                if (i != k):
                    train  = train  + folds[i]
            k_folds.append([test,train])
        elif(k == 1):
            for i in range(0, len(folds)):
                if (i != k):
                    train  = train  + folds[i]
            k_folds.append([test,train])
        elif(k == 2):
            for i in range(0, len(folds)):
                if (i != k):
                    train  = train  + folds[i]
            k_folds.append([test,train])
        elif(k == 3):
            for i in range(0, len(folds)):
                if (i != k):
                    train  = train  + folds[i]
            k_folds.append([test,train])
        elif(k == 4):
            for i in range(0, len(folds)):
                if (i != k):
                    train  = train  + folds[i]
            k_folds.append([test,train])    

    return k_folds

# cross_validation_: Return testing set=25%, training set=75%
def cross_validation_(matrix, n_users):
    users = []
    for u in range (0, n_users):
        users.append((matrix[u].nnz, u))
    users.sort()
    users = users[::-1]

    fold_test = []
    fold_train = []
    
    k = 4
    # se construyen los fold de manera uniforme.
    for i, (n_ratings, u) in enumerate(users):
        if(i%k == 0):
            fold_test.append(u)
        elif(i%k == 1):
            fold_train.append(u)
        elif(i%k == 2):
            fold_train.append(u)
        elif(i%k == 3):
            fold_train.append(u)

    return fold_test, fold_train

def _similitud_orden_superior(u_a,v,productos_train,u_productos_train,m_ratings,user_test,user_train,
                             good_users,u_a_similarities,min_intersect,user_exp,type='coseno'):
    if(type == 'coseno'):
        similarities = []
        for v_j in productos_train:
            # Calcular sim(v,v_j)
            sim_vj = cosine_similarity(m_ratings[user_train, v].toarray().T, m_ratings[user_train, v_j].toarray().T)[0][0]
            # Solo usuarios que esten en el conjunto train y su rating no sea nulo en v_j
            usuarios_segundo_orden = np.intersect1d(user_train, m_ratings[:, v_j].nonzero()[0])
            for u_i in usuarios_segundo_orden:
                i_min=1
                u_exp=0
                if(min_intersect>1):
                    i_min=len(np.intersect1d(m_ratings[u_i].nonzero()[1], productos_train))
                if(user_exp>0):
                    u_exp=m_ratings[u_i].nnz
                if(u_i in good_users and i_min >= min_intersect and u_exp >= user_exp):
                    # Calcular sim(u_a, u_i)
                    if(u_a_similarities[u_i] == -2):
                        if(u_i != u_a):
                            sim_ui = cosine_similarity([u_productos_train],  [m_ratings[u_i].toarray()[0]])[0][0]
                            similarities.append((u_i, sim_ui, v_j, sim_vj))
                            u_a_similarities[u_i] = sim_ui
                        else:
                            similarities.append((u_i, 0, v_j, sim_vj))
                    else:
                        similarities.append((u_i, u_a_similarities[u_i], v_j, sim_vj))
    elif(type == 'pearson'):
        similarities = []
        for v_j in productos_train:
            # Calcular sim(v,v_j)
            sim_vj = _pearsonr(m_ratings[user_train,v].toarray().T[0], m_ratings[user_train,v_j].toarray().T[0])
            # Solo usuarios que esten en el conjunto train 
            usuarios_segundo_orden = np.intersect1d(user_train, m_ratings[:, v_j].nonzero()[0])
            for u_i in usuarios_segundo_orden:
                i_min=1
                u_exp=0
                if(min_intersect>1):
                    i_min=len(np.intersect1d(m_ratings[u_i].nonzero()[1], productos_train))
                if(user_exp>0):
                    u_exp=m_ratings[u_i].nnz
                # Solo usuarios que pertenezcan a good_users y que tengan una interseccion minima (|I(u) intersect I(u_i)| >= 2)
                if(u_i in good_users and i_min >= min_intersect and u_exp >= user_exp):
                    # Calcular sim(u_a, u_i)
                    if(u_a_similarities[u_i] == -2):
                        if(u_i != u_a):
                            sim_ui = _pearsonr(u_productos_train,  m_ratings[u_i].toarray()[0])
                            similarities.append((u_i, sim_ui, v_j, sim_vj))
                            u_a_similarities[u_i] = sim_ui
                        else:
                            similarities.append((u_i, 0, v_j, sim_vj))
                    else:
                        similarities.append((u_i, u_a_similarities[u_i], v_j, sim_vj))
    return similarities

def _good_users(n_users, users_mean):
    percentil = np.percentile(users_mean, [33,66])

    pesimista = []
    neutro = []
    optimista = []
    for u in range(0, n_users):
        if(users_mean[u] <= percentil[0]):
            pesimista.append(u)
        elif(users_mean[u] > percentil[0] and  users_mean[u] <= percentil[1]):
            neutro.append(u)
        elif(users_mean[u] > percentil[1]):
            optimista.append(u)

    return pesimista, neutro, optimista

def entropia_(prediccion,movies,genres):
    generos = []
    for i, (v, r) in enumerate(prediccion):
        gen=[]
        g = movies.loc[v, genres].tolist()
        for i in range(0, len(g)):
            if g[i] == 1:
                gen.append(i)
        generos.append(gen)
    clases= {0:0., 1:0., 2:0., 3:0., 4:0., 5:0., 6:0., 7:0., 8:0., 9:0., 10:0., 11:0., 12:0., 13:0., 14:0., 15:0., 16:0., 17:0., 18:0.}
    m_i = 0.0
    for g_ in generos:
        m_i += len(g_)
        for i in g_:
            clases[i]+=1.
    c_dis = 0
    for key, value in clases.items():
        if value !=0:
            c_dis += 1
    e_i = 0.0    
    for key, value in clases.items():
        if value !=0:
            if(value/m_i == 1):
                e_i+=0
            else:
                e_i+= value/m_i*log(value/m_i,c_dis)
                #e_i+= value/m_i*log(value/m_i,2)
    return -1*e_i