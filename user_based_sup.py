import random
import numpy as np
from time import time
from funciones import *
from funciones import _precision_recall, _similitud_orden_superior
# N = Top-N list
# x =  cantidad de ratings calificados (vistos) por los usuarios de test
# good_rating = Ratings relevantes (umbral)
# Seleccionar de manera random x = 30 peliculas para cada usuario considarando que se quiere realizar un N-top list,
# cada usuario debe almenos tener x + N peliculas
def user_based_CV(m_ratings, n_users, n_items, users_mean, users_std, good_users, x, N,
                 good_rating, min_int, u_exp, norm_expansion = 'none', similitud = 'coseno'):
    tiempo_total_ini = time()
    k_folds = k_fold_(m_ratings, n_users)

    total_precision_at_k = []
    total_recall_at_k = []
    total_f_score_at_k = []
    for k, (u_test, u_train) in enumerate(k_folds):
        productos_mean = np.zeros(n_items)
        productos_std = np.zeros(n_items)

        for v in range(0, n_items):
            producto_ratings = m_ratings[u_train, v][m_ratings[u_train, v].nonzero()[0]].toarray().T[0]
            if(len(producto_ratings)==0):
                productos_mean[v] = 0
                productos_std[v] = 0
            else:
                productos_mean[v] = np.mean(producto_ratings)
                productos_std[v] = np.std(producto_ratings)
        
        tiempo_ini = time()
        precision_at_k = []
        recall_at_k = []
        for u in u_test:
            if(m_ratings[u].nnz > x + N):
                index_rand_productos = random.sample(xrange(m_ratings[u].nnz), x)
                productos_train = m_ratings[u].nonzero()[1][index_rand_productos]
                productos_test = np.setdiff1d(m_ratings[u].nonzero()[1], productos_train)
                ratings_test = m_ratings[u, productos_test].toarray()[0]
                # productos_relevantes: productos relevantes del conjunto productos_test (productos a recomendar)
                productos_relevantes = productos_test[np.where(ratings_test >= good_rating)[0]]

                # Verificamos que almenos un producto a recomendar sea relevante, de lo contrario no tiene sentido
                # calcular las metricas
                if(len(productos_relevantes) > 0):
                    # Se calcula el promedio y la std para el usuario respecto a las x productos que se utilizaran
                    u_mean, u_std = one_user_mean_std(m_ratings[u, productos_train].toarray()[0])

                    # Se crea un vector de zeros para almacenar solo los x ratings que se pueden usar para el calculo de 
                    # similitud del usuario objetivo
                    u_productos = np.zeros(n_items)
                    u_productos[productos_train] = m_ratings[u, productos_train].toarray()[0]
                    # Se realiza la prediccion de ratings para el usuario
                    pred = _user_based(u,u_productos,n_users,n_items,productos_test,productos_train,m_ratings,
                                        u_test,u_train,u_mean,u_std,users_mean,users_std,productos_mean,
                                        productos_std,good_users,min_int,u_exp,norm_expansion,similarity=similitud)
                    #Se calcula la precision y recall para las recomendaciones de u
                    _precision_recall(pred, productos_relevantes, N, precision_at_k, recall_at_k)

        total_precision_at_k.append(np.mean(precision_at_k, axis = 0))
        total_recall_at_k.append(np.mean(recall_at_k, axis = 0))
        tiempo_fin = time()
        print 'Tiempo de user_based para un k-fold: ',(tiempo_fin - tiempo_ini)
        break

    #Se calculan las metricas promediando las obtenidas en cada k-fold
    P_at_k = np.mean(total_precision_at_k, axis = 0)
    R_at_k = np.mean(total_recall_at_k, axis = 0)
    F1_at_k = 2*np.multiply(P_at_k, R_at_k) / (P_at_k + R_at_k)

    tiempo_total_fin = time()
    print 'Tiempo total para 5-fold: ',(tiempo_total_fin - tiempo_total_ini)
    
    return P_at_k, R_at_k, F1_at_k


def _user_based(u_a,u_productos_train,n_users,n_items,productos_test,productos_train,m_ratings,user_test,
                user_train,u_a_mean,u_a_std,users_mean,users_std,items_mean,items_std,
                good_users,min_int,u_exp,norm_exp,similarity='coseno'):
    pred = []
    # Arreglo auxiliar para no re-calcular las similitudes que ya fueron computadas
    u_a_similarities = -2*np.ones(n_users)
    # productos: son los articulos que se desean recomendar
    for v in productos_test:
        # Productos calificados por el usuario en cold start
        num = 0.0
        den = 0.0
        similitudes=_similitud_orden_superior(u_a,v,productos_train,u_productos_train,m_ratings,user_test,
                                                user_train,good_users,u_a_similarities,min_int,u_exp,type=similarity)
        #calcular sim(u_a, u_i) * R(u_i, v_j) * Sim(v_j, v)
        for i, (u_i, sim_ui, v_j, sim_vj) in enumerate(similitudes):
            if(norm_exp == 'none'):
                num = num + sim_ui * m_ratings[u_i, v_j] * sim_vj
            elif(norm_exp == 'mean_centering_v'):
                num = num + sim_ui * (m_ratings[u_i, v_j] - items_mean[v_j]) * sim_vj
            elif(norm_exp == 'mean_centering_u'):
                num = num + sim_ui * (m_ratings[u_i, v_j] - users_mean[u_i]) * sim_vj
            elif(norm_exp == 'z_score_v'):
                num = num + sim_ui * ((m_ratings[u_i, v_j] - items_mean[v_j]) / items_std[v_j]) * sim_vj
            elif(norm_exp == 'z_score_u'):
                num = num + sim_ui * ((m_ratings[u_i, v_j] - users_mean[u_i]) / users_std[u_i]) * sim_vj
            den = den + abs(sim_ui * sim_vj)
        if den > 0:
            if(norm_exp == 'none'):
                pred.append((v, num/den))
            elif(norm_exp == 'mean_centering_v'):
                pred.append((v, items_mean[v] + num/den))
            elif(norm_exp == 'mean_centering_u'):
                pred.append((v, u_a_mean + num/den))
            elif(norm_exp == 'z_score_v'):
                pred.append((v, items_mean[v] + (items_std[v]*num/den) ))
            elif(norm_exp == 'z_score_u'):
                pred.append((v, u_a_mean + (u_a_std * num/den)))
        elif den == 0:
            pred.append((v, 0))
    return pred