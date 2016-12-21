#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots import *
from funciones import *
from user_based import user_based_CV,user_based_CV_ndcg, _user_based
from time import time
from scipy.sparse import csr_matrix
from scipy.io import mmread

title_genres = ['title','unknown', 'Action', 'Adventure', 'Animation','Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
          'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery','Romance','Sci-Fi', 'Thriller', 'War', 'Western']

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
          'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
          'Romance','Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(24), encoding='latin-1')

#m_cols = ['movie_id', 'title', 'genres']
#movies = pd.read_csv('ml-1m/movies.dat', sep='::', names=m_cols, engine='python')
#movies = pd.read_csv('ml-10m/movies.dat', sep='::', names=m_cols, engine='python')
#movies = pd.read_csv('ml-20m/movies.csv', sep=',', names=m_cols)

hash_v = hash_productos(movies, 'movie_id')

# Lectura y almacenamiento de la matriz de ratings dispersa
ratings = csr_matrix(mmread('ml-100k/ratings.mtx'))

# Filtrado de productos que no poseen calificaciones (en caso de existir)
m_ratings = ratings[:,hash_v.values()]

# Item based: Transponer la matriz de ratings
#m_ratings = m_ratings.T

n_users = m_ratings.shape[0]
n_items = m_ratings.shape[1]
print(u'Número de usuarios = {} | Número de productos = {}'.format(n_users, n_items))

#Calculo de la desviacion estandar y el promedio para cada usuario
tiempo_ini = time()
users_mean, users_std = user_mean_std(m_ratings, n_users)
tiempo_fin = time()

N=int(sys.argv[4])
if(sys.argv[2] == 'umbral_relevancia'):
    goodRating = [3,4,5]
    metricas = []
    for g_r in goodRating:
        p_at_k,r_at_k,f_at_k,var=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,x=20,N=N,
            good_rating=g_r,k_max=40,normalizacion='none',similitud='pearson')
        metricas.append((g_r,p_at_k,r_at_k))
    plot_umbral_relevancia(goodRating, metricas)
elif(sys.argv[2]=='varianza'):
    metricas = []
    sim = ['coseno', 'pearson']
    for s in sim:
        p_at_k,r_at_k,f_at_k,var=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,x=20,N=N,
            good_rating=4,k_max=40,normalizacion='none',similitud=s)
        metricas.append((s,p_at_k,r_at_k,f_at_k,var))

    plot_varianza(metricas,N)
elif(sys.argv[2]=='tamanno_vecindario'):
    metricas = []
    vecindario_max = np.linspace(20,400,20, dtype=int)
    sim = ['coseno', 'pearson']
    for s in sim:
        for k in vecindario_max:
            p_at_k,r_at_k,f_at_k,var=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,x=20,N=N,
                good_rating = 4,k_max=k,normalizacion='none',similitud = s)
            metricas.append((s,p_at_k, r_at_k, f_at_k))
    plot_tamanno_vecindario(metricas,vecindario_max)
elif(sys.argv[2]=='normalizacion'):
    metricas = []
    sim = ['coseno','pearson']
    norm = ['none','mean_centering','z_score']
    for s in sim:
        for n_ in norm:
            p_at_k,r_at_k,f_at_k,var=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,x=20,N=N,
                good_rating=4,k_max=40,normalizacion=n_,similitud=s)
            metricas.append((s,n_, p_at_k, r_at_k, f_at_k))
    plot_normalizacion(metricas,N)
elif(sys.argv[2]=='productos_recomendados'):
    metricas = []
    sim = ['coseno', 'pearson']
    for s in sim:
        p_at_k,r_at_k,f_at_k,var=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,x=20,N=N,
            good_rating=4,k_max=40,normalizacion='none',similitud=s)
        metricas.append((s,p_at_k,r_at_k,f_at_k))
    plot_productos_recomendados(metricas,N)
elif(sys.argv[2]=='trade_off'):
    metricas = []
    p_at_k,r_at_k,f_at_k,var=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,x=20,N=N,
        good_rating=4,k_max=40,normalizacion='none',similitud='pearson')
    metricas.append((p_at_k,r_at_k,f_at_k))
    plot_trade_off(metricas,N)
elif(sys.argv[2]=='cold_start'):
    metricas = []
    sim = ['coseno', 'pearson']
    given = [2,3,5,8,10,15,20]
    for s in sim:
        for g in given:
            p_at_k,r_at_k,f_at_k,var=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,x=g,N=N,
                good_rating=4,k_max=40,normalizacion='none',similitud=s)
            metricas.append((s,p_at_k, r_at_k, f_at_k))
    plot_cold_start(metricas,given)
elif(sys.argv[2]=='ndcg'):
    metricas = []
    sim = ['coseno', 'pearson']
    for s in sim:
        nDCG=user_based_CV_ndcg(m_ratings,n_users,n_items,users_mean,users_std,x=20,N=N,good_rating=4,
                            k_max=40,normalizacion='none',similitud=s)
        metricas.append((s,nDCG))
    plot_ndcg(metricas,N)