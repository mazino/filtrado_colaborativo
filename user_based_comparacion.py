import random
import numpy as np
from time import time
from funciones import *
from funciones import _similitud,_similitud_orden_superior,_precision_recall,_nDCG,entropia_


def user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,good_users,x,N,good_rating,
                  k_max,alpha,min_int,u_exp,norm_expansion,normalizacion='none',similitud='coseno'):
    tiempo_total_ini = time()
    k_folds = k_fold_(m_ratings, n_users)

    total_precision_at_k = []
    total_recall_at_k = []
    total_f_score_at_k = []

    total_precision_at_k_sup = []
    total_recall_at_k_sup = []
    total_f_score_at_k_sup = []
    
    total_precision_at_k_hibrido = []
    total_recall_at_k_hibrido = []
    total_f_score_at_k_hibrido = []

    for k, (u_test, u_train) in enumerate(k_folds):
        tiempo_ini = time()

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

        precision_at_k = []
        recall_at_k = []

        precision_at_k_sup = []
        recall_at_k_sup = []

        precision_at_k_hibrido = []
        recall_at_k_hibrido = []

        for u in u_test:
            if(m_ratings[u].nnz > x + N):
                # realizar un random de 0..m_ratings[u].nnz (cantidad de ratings no nulos) de x
                index_rand_productos = random.sample(xrange(m_ratings[u].nnz), x)
                # Se seleccionan los productos calificados por u y de estos se obtiene los x=30 productos (indices)
                # visibles del test data
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

                    #productos_train.sort()

                    # Se crea un vector de zeros para almacenar solo los x=5 ratings que se pueden usar para el calculo de 
                    # similitud del usuario objetivo
                    u_productos = np.zeros(n_items)
                    u_productos[productos_train] = m_ratings[u, productos_train].toarray()[0]

                    # Se realiza la prediccion de ratings para el usuario
                    pred,pred_ord_sup,pred_hibrido=_user_based(u,u_productos,n_users,n_items,productos_test,
                                productos_train,m_ratings,u_test,u_train,u_mean,u_std,users_mean,users_std,
                                productos_mean,productos_std,good_users,k_max,alpha,min_int,u_exp,
                                norm_expansion,type=normalizacion,similarity=similitud)

                    #Se calcula la precision y recall para las recomendaciones de u
                    _precision_recall(pred, productos_relevantes, N, precision_at_k, recall_at_k)
                    _precision_recall(pred_ord_sup, productos_relevantes, N, precision_at_k_sup, recall_at_k_sup)
                    _precision_recall(pred_hibrido, productos_relevantes, N, precision_at_k_hibrido, recall_at_k_hibrido)

        total_precision_at_k.append(np.mean(precision_at_k, axis = 0))
        total_recall_at_k.append(np.mean(recall_at_k, axis = 0))
        
        total_precision_at_k_sup.append(np.mean(precision_at_k_sup, axis = 0))
        total_recall_at_k_sup.append(np.mean(recall_at_k_sup, axis = 0))
        
        total_precision_at_k_hibrido.append(np.mean(precision_at_k_hibrido, axis = 0))
        total_recall_at_k_hibrido.append(np.mean(recall_at_k_hibrido, axis = 0))

        tiempo_fin = time()
        print 'Tiempo para un fold: ',(tiempo_fin - tiempo_ini)

    #Se calculan las metricas promediando las obtenidas en cada k-fold
    P_at_k = np.mean(total_precision_at_k, axis = 0)
    R_at_k = np.mean(total_recall_at_k, axis = 0)
    F1_at_k = 2*np.multiply(P_at_k, R_at_k) / (P_at_k + R_at_k)
    
    P_at_k_sup = np.mean(total_precision_at_k_sup, axis = 0)
    R_at_k_sup = np.mean(total_recall_at_k_sup, axis = 0)
    F1_at_k_sup = 2*np.multiply(P_at_k_sup, R_at_k_sup) / (P_at_k_sup + R_at_k_sup)
    
    P_at_k_hib = np.mean(total_precision_at_k_hibrido, axis = 0)
    R_at_k_hib = np.mean(total_recall_at_k_hibrido, axis = 0)
    F1_at_k_hib = 2*np.multiply(P_at_k_hib, R_at_k_hib) / (P_at_k_hib + R_at_k_hib)

    tiempo_total_fin = time()
    print 'Tiempo total para 5-fold: ',(tiempo_total_fin - tiempo_total_ini)
    
    return P_at_k, R_at_k, F1_at_k, P_at_k_sup, R_at_k_sup, F1_at_k_sup, P_at_k_hib, R_at_k_hib, F1_at_k_hib
def user_based_ndcg(m_ratings,n_users,n_items,users_mean,users_std,good_users,x,N,good_rating,
                    k_max,alpha,norm_expansion,min_int,u_exp,normalizacion='none',similitud='coseno'):
    tiempo_total_ini = time()
    k_folds = k_fold_(m_ratings, n_users)

    total_nDCG_at_k = []
    total_nDCG_at_k_sup = []
    total_nDCG_at_k_hibrido = []

    for k, (u_test, u_train) in enumerate(k_folds):
        tiempo_ini = time()
        
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

        nDCG_at_k = []
        nDCG_at_k_sup = []
        nDCG_at_k_hibrido = []
        for u in u_test:
            if(m_ratings[u].nnz > x + N):
                index_rand_productos = random.sample(xrange(m_ratings[u].nnz), x)
                productos_train = m_ratings[u].nonzero()[1][index_rand_productos]
                productos_test = np.setdiff1d(m_ratings[u].nonzero()[1], productos_train)

                u_mean, u_std = one_user_mean_std(m_ratings[u, productos_train].toarray()[0])

                u_productos = np.zeros(n_items)
                u_productos[productos_train] = m_ratings[u, productos_train].toarray()[0]

                pred,pred_ord_sup,pred_hibrido=_user_based(u,u_productos,n_users,n_items,productos_test,
                            productos_train, m_ratings,u_test,u_train,u_mean,u_std,users_mean,users_std,
                            productos_mean,productos_std,good_users,k_max,alpha,min_int,u_exp,
                            norm_expansion,type=normalizacion,similarity=similitud)

                #Se calcula nDCG para las recomendaciones de u
                _nDCG(u, pred, m_ratings, N, nDCG_at_k)
                _nDCG(u, pred_ord_sup, m_ratings, N, nDCG_at_k_sup)
                _nDCG(u, pred_hibrido, m_ratings, N, nDCG_at_k_hibrido)

        total_nDCG_at_k.append(np.mean(nDCG_at_k, axis = 0))
        total_nDCG_at_k_sup.append(np.mean(nDCG_at_k_sup, axis = 0))
        total_nDCG_at_k_hibrido.append(np.mean(nDCG_at_k_hibrido, axis = 0))

        tiempo_fin = time()
        print 'Tiempo para un fold : ',(tiempo_fin - tiempo_ini)

    #Se calculan las metricas promediando las obtenidas en cada k-fold
    nDCG = np.mean(total_nDCG_at_k, axis = 0)
    nDCG_sup = np.mean(total_nDCG_at_k_sup, axis = 0)
    nDCG_hib = np.mean(total_nDCG_at_k_hibrido, axis = 0)

    tiempo_total_fin = time()
    print 'Tiempo total para 5-fold: ',(tiempo_total_fin - tiempo_total_ini)

    return nDCG, nDCG_sup, nDCG_hib
def user_based_entropia(m_ratings,movies,genres,n_users,n_items,users_mean,users_std,good_users,x,N,good_rating,
                    k_max,alpha,norm_expansion,min_int,u_exp,normalizacion='none',similitud='coseno'):
    tiempo_total_ini = time()
    k_folds = k_fold_(m_ratings, n_users)

    total_entropia_at_k=[]
    total_entropia_at_k_sup=[]

    interseccion_v=[]
    jaccard_v=[]

    for k, (u_test, u_train) in enumerate(k_folds):
        tiempo_ini = time()

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

        entropia_at_k=[] = []
        entropia_at_k_sup = []
        
        interseccion_productos = []
        jaccard_productos = []
        
        for u in u_test:
            if(m_ratings[u].nnz > x + N):
                index_rand_productos = random.sample(xrange(m_ratings[u].nnz), x)
                productos_train = m_ratings[u].nonzero()[1][index_rand_productos]
                productos_test = np.setdiff1d(m_ratings[u].nonzero()[1], productos_train)

                u_mean, u_std = one_user_mean_std(m_ratings[u, productos_train].toarray()[0])

                # Se crea un vector de zeros para almacenar solo los x=5 ratings que se pueden usar para el calculo de 
                # similitud del usuario objetivo
                u_productos = np.zeros(n_items)
                u_productos[productos_train] = m_ratings[u, productos_train].toarray()[0]

                # Se realiza la prediccion de ratings para el usuario
                pred,pred_ord_sup,pred_hibrido=_user_based(u,u_productos,n_users,n_items,productos_test,
                            productos_train,m_ratings,u_test,u_train,u_mean,u_std,users_mean,users_std,
                            productos_mean,productos_std,good_users,k_max,alpha,min_int,u_exp,
                            norm_expansion,type=normalizacion,similarity=similitud)

                #Se calcula la entropia de la recomendacion
                pred.sort(key=lambda tup: tup[1], reverse=True)
                pred_ord_sup.sort(key=lambda tup: tup[1], reverse=True)

                entropia=[]
                entropia_sup=[]
                for i in range(1, N+1): #cambiar 11 por el largo de la lista (N + 1)
                    entropia.append(entropia_(pred[:i],movies,genres))
                    entropia_sup.append(entropia_(pred_ord_sup[:i],movies,genres))

                entropia_at_k.append(entropia)
                entropia_at_k_sup.append(entropia_sup)
                
                #Interseccion de productos y Jaccard
                prod,rat= zip(*pred)
                prod_sup,rat_sup= zip(*pred_ord_sup)

                interseccion = [float(len(np.intersect1d(prod[:10], prod_sup[:10]))), float(len(np.intersect1d(prod[:20], prod_sup[:20])))]
                union = [len(np.union1d(prod[:10], prod_sup[:10])),len(np.union1d(prod[:20], prod_sup[:20]))]

                interseccion_productos.append(interseccion)
                jaccard_productos.append(np.divide(interseccion,union))

        total_entropia_at_k.append(np.mean(entropia_at_k, axis = 0))
        total_entropia_at_k_sup.append(np.mean(entropia_at_k_sup, axis = 0))

        interseccion_v.append(np.mean(interseccion_productos, axis = 0))
        jaccard_v.append(np.mean(jaccard_productos, axis = 0))

        tiempo_fin = time()
        print 'Tiempo para un k-fold: ',(tiempo_fin - tiempo_ini)

    #Se calculan las metricas promediando las obtenidas en cada k-fold
    Entropia = np.mean(total_entropia_at_k, axis = 0)
    Entropia_sup = np.mean(total_entropia_at_k_sup, axis = 0)

    interseccion_v_total = np.mean(interseccion_v, axis = 0)
    jaccard_v_total = np.mean(jaccard_v, axis = 0)

    tiempo_total_fin = time()
    print 'Tiempo para 5-fold: ',(tiempo_total_fin - tiempo_total_ini)
    
    return Entropia, Entropia_sup, interseccion_v_total, jaccard_v_total
def _user_based(u_a,u_productos_train,n_users,n_items,productos_test,productos_train,m_ratings,user_test,user_train,
                u_a_mean,u_a_std,users_mean,users_std,items_mean,items_std,good_users,k_max,alpha,min_int,u_exp,
                norm_exp,type='none',similarity='coseno'):
    pred = []
    pred_ord_sup = []
    pred_hibrido = []
    # Arreglo auxiliar para no re-calcular las similitudes que ya fueron computadas
    u_a_similarities = -2*np.ones(n_users)
    # productos: son los articulos que se desean recomendar
    for v in productos_test:
        num = 0
        den = 0
        #Similitud entre u_a y los usuarios inducidos por el producto v
        user_similarities = _similitud(u_a, v, u_productos_train, m_ratings, user_test, u_a_similarities, type = similarity)
        # generamos el vecindario de los k usuarios que han visto v
        vecindario = k_neighborhood(user_similarities, k_max)
        # Vecindario es una tupla (usuario, similitud) u = user, sim = similitud con u_a
        if (type == 'none' ):
            for i, (u, sim) in enumerate(vecindario):
                num = num + sim*m_ratings[u,v]
                den = den + abs(sim)
            if den > 0:
                prediccion = num/den
            elif den == 0:
                prediccion = 0
        elif(type == 'mean_centering'):
            for i, (u, sim) in enumerate(vecindario):
                num = num + (sim * (m_ratings[u,v] - users_mean[u]))
                den = den + abs(sim)
            if den > 0:
                prediccion = u_a_mean + num/den
            elif den == 0:
                prediccion = 0
        elif(type == 'z_score'):
            for i, (u, sim) in enumerate(vecindario):
                num = num + ( sim * (m_ratings[u,v] - users_mean[u]) / users_std[u] )
                den = den + abs(sim)
            if den > 0:
                prediccion = u_a_mean + (u_a_std * num/den)
            elif den == 0:
                prediccion = 0

        num = 0
        den = 0
        user_similarities=_similitud_orden_superior(u_a,v,productos_train,u_productos_train,m_ratings,user_test,
                                                user_train,good_users,u_a_similarities,min_int,u_exp,type=similarity)
        #calcular sim(u_a, u_i) * R(u_i, v_j) * Sim(v_j, v)
        for i, (u_i, sim_ui, v_j, sim_vj) in enumerate(user_similarities):
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
                prediccion_orden_sup = num/den
            elif(norm_exp == 'mean_centering_v'):
                prediccion_orden_sup = items_mean[v] + num/den
            elif(norm_exp == 'mean_centering_u'):
                prediccion_orden_sup = u_a_mean + num/den
            elif(norm_exp == 'z_score_v'):
                prediccion_orden_sup = items_mean[v] + (items_std[v]*num/den)
            elif(norm_exp == 'z_score_u'):
                prediccion_orden_sup = u_a_mean + (u_a_std * num/den)
        elif den == 0:
            prediccion_orden_sup = 0

        pred.append((v, prediccion))
        pred_ord_sup.append((v, prediccion_orden_sup))
        pred_hibrido.append((v, alpha * prediccion_orden_sup + (1 - alpha) * prediccion ))

    return pred, pred_ord_sup, pred_hibrido

#################### funciones para 1M ####################
def user_based_CV_1M(m_ratings,n_users,n_items,users_mean,users_std,good_users,x,N,good_rating,
                  k_max,alpha,min_int,u_exp,norm_expansion,normalizacion='none',similitud='coseno'):
    tiempo_total_ini = time()
    u_test, u_train=cross_validation_(m_ratings,n_users)

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

    precision_at_k = []
    recall_at_k = []

    precision_at_k_sup = []
    recall_at_k_sup = []

    precision_at_k_hibrido = []
    recall_at_k_hibrido = []

    for u in u_test:
        if(m_ratings[u].nnz > x + N):
            # realizar un random de 0..m_ratings[u].nnz (cantidad de ratings no nulos) de x
            index_rand_productos = random.sample(xrange(m_ratings[u].nnz), x)
            # Se seleccionan los productos calificados por u y de estos se obtiene los x=30 productos (indices)
            # visibles del test data
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

                #productos_train.sort()

                # Se crea un vector de zeros para almacenar solo los x=5 ratings que se pueden usar para el calculo de 
                # similitud del usuario objetivo
                u_productos = np.zeros(n_items)
                u_productos[productos_train] = m_ratings[u, productos_train].toarray()[0]

                # Se realiza la prediccion de ratings para el usuario
                pred,pred_ord_sup,pred_hibrido=_user_based(u,u_productos,n_users,n_items,productos_test,
                            productos_train,m_ratings,u_test,u_train,u_mean,u_std,users_mean,users_std,
                            productos_mean,productos_std,good_users,k_max,alpha,min_int,u_exp,
                            norm_expansion,type=normalizacion,similarity=similitud)

                #Se calcula la precision y recall para las recomendaciones de u
                _precision_recall(pred, productos_relevantes, N, precision_at_k, recall_at_k)
                _precision_recall(pred_ord_sup, productos_relevantes, N, precision_at_k_sup, recall_at_k_sup)
                _precision_recall(pred_hibrido, productos_relevantes, N, precision_at_k_hibrido, recall_at_k_hibrido)

    P_at_k = np.mean(precision_at_k, axis = 0)
    R_at_k = np.mean(recall_at_k, axis = 0)
    F1_at_k = 2*np.multiply(P_at_k, R_at_k) / (P_at_k + R_at_k)
    
    P_at_k_sup = np.mean(precision_at_k_sup, axis = 0)
    R_at_k_sup = np.mean(recall_at_k_sup, axis = 0)
    F1_at_k_sup = 2*np.multiply(P_at_k_sup, R_at_k_sup) / (P_at_k_sup + R_at_k_sup)
    
    P_at_k_hib = np.mean(precision_at_k_hibrido, axis = 0)
    R_at_k_hib = np.mean(recall_at_k_hibrido, axis = 0)
    F1_at_k_hib = 2*np.multiply(P_at_k_hib, R_at_k_hib) / (P_at_k_hib + R_at_k_hib)

    tiempo_total_fin = time()
    print 'Tiempo total: ',(tiempo_total_fin - tiempo_total_ini)
    
    return P_at_k, R_at_k, F1_at_k, P_at_k_sup, R_at_k_sup, F1_at_k_sup, P_at_k_hib, R_at_k_hib, F1_at_k_hib
def user_based_ndcg_1M(m_ratings,n_users,n_items,users_mean,users_std,good_users,x,N,good_rating,
                    k_max,alpha,norm_expansion,min_int,u_exp,normalizacion='none',similitud='coseno'):
    tiempo_total_ini = time()
    u_test, u_train=cross_validation_(m_ratings,n_users)

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

    nDCG_at_k = []
    nDCG_at_k_sup = []
    nDCG_at_k_hibrido = []
    for u in u_test:
        if(m_ratings[u].nnz > x + N):
            index_rand_productos = random.sample(xrange(m_ratings[u].nnz), x)
            productos_train = m_ratings[u].nonzero()[1][index_rand_productos]
            productos_test = np.setdiff1d(m_ratings[u].nonzero()[1], productos_train)

            u_mean, u_std = one_user_mean_std(m_ratings[u, productos_train].toarray()[0])

            u_productos = np.zeros(n_items)
            u_productos[productos_train] = m_ratings[u, productos_train].toarray()[0]

            pred,pred_ord_sup,pred_hibrido=_user_based(u,u_productos,n_users,n_items,productos_test,
                        productos_train, m_ratings,u_test,u_train,u_mean,u_std,users_mean,users_std,
                        productos_mean,productos_std,good_users,k_max,alpha,min_int,u_exp,
                        norm_expansion,type=normalizacion,similarity=similitud)

            #Se calcula nDCG para las recomendaciones de u
            _nDCG(u, pred, m_ratings, N, nDCG_at_k)
            _nDCG(u, pred_ord_sup, m_ratings, N, nDCG_at_k_sup)
            _nDCG(u, pred_hibrido, m_ratings, N, nDCG_at_k_hibrido)

    nDCG = np.mean(nDCG_at_k, axis = 0)
    nDCG_sup = np.mean(nDCG_at_k_sup, axis = 0)
    nDCG_hib = np.mean(nDCG_at_k_hibrido, axis = 0)

    tiempo_total_fin = time()
    print 'Tiempo total: ',(tiempo_total_fin - tiempo_total_ini)

    return nDCG, nDCG_sup, nDCG_hib