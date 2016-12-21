import random
import numpy as np
from time import time
from funciones import *
from funciones import _similitud, _precision_recall, _nDCG

def user_based_CV(m_ratings, n_users, n_items, users_mean, users_std, x, N,
                 good_rating, k_max, normalizacion, similitud):
    tiempo_total_ini = time()
    #k-fold-cross-validation
    k_folds = k_fold_(m_ratings, n_users)

    total_precision_at_k = []
    total_recall_at_k = []
    total_f_score_at_k = []
    for k, (u_test, u_train) in enumerate(k_folds):
        tiempo_ini = time()
        # Se declara una lista para almacenar todas las precisiones y recalls de cada usuario del conjunto de test
        precision_at_k = []
        recall_at_k = []
        for u in u_test:
            if(m_ratings[u].nnz > x + N):
                # realizar un random de 0..m_ratings[u].nnz (cantidad de ratings no nulos) de x
                index_rand_productos = random.sample(xrange(m_ratings[u].nnz), x)
                # Se seleccionan los productos calificados por u y de estos se obtiene los x=30 productos (indices)
                # visibles del conjunto de test
                # productos_train: son los productos de u con ratings visibles
                productos_train = m_ratings[u].nonzero()[1][index_rand_productos]
                # Peliculas que se deben recomendar (complemento de las peliculas vistas seleccionadas al azar)
                # Productos_test: son los productos que se debe recomendar
                productos_test = np.setdiff1d(m_ratings[u].nonzero()[1], productos_train)
                # ratings del conjunto que se recomienda
                ratings_test = m_ratings[u, productos_test].toarray()[0]
                # productos_relevantes: productos relevantes del conjunto productos_test (productos a recomendar)
                productos_relevantes = productos_test[np.where(ratings_test >= good_rating)[0]]

                # Verificamos que almenos un producto a recomendar sea relevante, de lo contrario no tiene sentido
                # calcular las medidas
                if(len(productos_relevantes) > 0):
                    # Se calcula el promedio y la std para el usuario respecto a las 30 productos que se utilizaran
                    u_mean, u_std = one_user_mean_std(m_ratings[u, productos_train].toarray()[0])

                    # Se crea un vector de zeros para almacenar solo los x=30 ratings que se pueden usar para el calculo de 
                    # similitud del usuario objetivo
                    u_productos = np.zeros(n_items)
                    u_productos[productos_train] = m_ratings[u, productos_train].toarray()[0]

                    # Se realiza la prediccion de ratings para el usuario
                    pred=_user_based(u,u_productos,n_users,n_items,productos_test,m_ratings,u_test,u_mean,u_std,
                                    users_mean,users_std,k_max,type=normalizacion,similarity=similitud)
                    #Se calcula la precision y recall para las recomendaciones de u
                    _precision_recall(pred, productos_relevantes, N, precision_at_k, recall_at_k)
        total_precision_at_k.append(np.mean(precision_at_k, axis = 0))
        total_recall_at_k.append(np.mean(recall_at_k, axis = 0))
        tiempo_fin = time()
        print 'Tiempo per k-fold: ',(tiempo_fin - tiempo_ini)

    #Se calculan las metricas promediando las obtenidas en cada k-fold
    P_var_fold = np.var(total_precision_at_k, axis=0)
    P_at_k = np.mean(total_precision_at_k, axis = 0)
    R_at_k = np.mean(total_recall_at_k, axis = 0)
    F1_at_k = 2*np.multiply(P_at_k, R_at_k) / (P_at_k + R_at_k)

    tiempo_total_fin = time()
    print 'Tiempo total para 5-fold: ',(tiempo_total_fin - tiempo_total_ini)
    
    return P_at_k, R_at_k, F1_at_k, P_var_fold

def user_based_CV_ndcg(m_ratings,n_users,n_items,users_mean,users_std,x,N,
                        good_rating,k_max,normalizacion,similitud):
    tiempo_total_ini = time()
    k_folds = k_fold_(m_ratings, n_users)

    total_nDCG_at_k = []
    for k, (u_test, u_train) in enumerate(k_folds):
        tiempo_ini = time()
        nDCG_at_k = []
        for u in u_test:
            if(m_ratings[u].nnz > x + N):
                index_rand_productos = random.sample(xrange(m_ratings[u].nnz), x)
                productos_train = m_ratings[u].nonzero()[1][index_rand_productos]
                productos_test = np.setdiff1d(m_ratings[u].nonzero()[1], productos_train)
               
                u_mean, u_std = one_user_mean_std(m_ratings[u, productos_train].toarray()[0])

                u_productos = np.zeros(n_items)
                u_productos[productos_train] = m_ratings[u, productos_train].toarray()[0]

                pred=_user_based(u,u_productos,n_users,n_items,productos_test,m_ratings,u_test,u_mean,u_std,
                                    users_mean,users_std,k_max,type=normalizacion,similarity=similitud)

                _nDCG(u, pred, m_ratings, N, nDCG_at_k)
        total_nDCG_at_k.append(np.mean(nDCG_at_k, axis = 0))
        tiempo_fin = time()
        print 'Tiempo de per k-fold: ',(tiempo_fin - tiempo_ini)
    nDCG = np.mean(total_nDCG_at_k, axis = 0)
    tiempo_total_fin = time()
    print 'Tiempo total para 5-fold: ',(tiempo_total_fin - tiempo_total_ini)

    return nDCG

# calculo de vecindario y prediccion
def _user_based(u_a, u_productos_train, n_users, n_items, productos_test, m_ratings, user_test,
                u_a_mean, u_a_std, users_mean, users_std, k_max, type = 'none', similarity = 'coseno'):
    pred = []
    # Arreglo auxiliar para no re-calcular las similitudes que ya fueron computadas
    u_similarities = -2*np.ones(n_users)
    if (type == 'none'):
        # productos: son los articulos que se desean recomendar
        for v in productos_test:
            num = 0
            den = 0
            #Similitud entre u_a y los usuarios inducidos por el producto v
            user_similarity = _similitud(u_a, v, u_productos_train, m_ratings, user_test, u_similarities, type = similarity)
            # generamos el vecindario de los k usuarios que han visto v
            vecindario = k_neighborhood(user_similarity, k_max)
            # Vecindario es una tupla (usuario, similitud) u = user, sim = similitud con u_a
            for i, (u, sim) in enumerate(vecindario):
                num = num + sim*m_ratings[u,v]
                den = den + abs(sim)
            if den > 0:
                pred.append((v, num/den))
            elif den == 0:
                pred.append((v, 0))
    elif(type == 'mean_centering'):
        for v in productos_test:
            num = 0
            den = 0
            user_similarity = _similitud(u_a, v, u_productos_train, m_ratings, user_test, u_similarities, type = similarity)
            vecindario = k_neighborhood(user_similarity, k_max)
            for i, (u, sim) in enumerate(vecindario):
                num = num + (sim * (m_ratings[u,v] - users_mean[u]))
                den = den + abs(sim)
            if den > 0:
                pred.append((v, u_a_mean + num/den))
            elif den == 0:
                pred.append((v, 0))
    elif(type == 'z_score'):
        for v in productos_test:
            num = 0
            den = 0
            user_similarity = _similitud(u_a, v, u_productos_train, m_ratings, user_test, u_similarities, type = similarity)
            vecindario = k_neighborhood(user_similarity, k_max)
            for i, (u, sim) in enumerate(vecindario):
                num = num + ( sim * (m_ratings[u,v] - users_mean[u]) / users_std[u] )
                den = den + abs(sim)
            if den > 0:
                pred.append( (v, u_a_mean + (u_a_std * num/den)) )
            elif den == 0:
                pred.append((v, 0))
    return pred