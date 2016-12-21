#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots import *
from funciones import *
from funciones import _good_users
from user_based_sup import user_based_CV, _user_based
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

pesimista, good_users, optimista = _good_users(n_users, users_mean)

N=int(sys.argv[4])
if(sys.argv[2] == 'expansion_segundo_orden'):
    metricas = []
    sim = ['coseno', 'pearson']
    for s in sim:
        p, r, f = user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,good_users,x=5,N=N,good_rating=4,
                                min_int=1,u_exp=0,norm_expansion='none',similitud=s)
        metricas.append((s,p,r,f))
    plot_expansion_segundo_orden(metricas,N)
elif(sys.argv[2]=='interseccion_minima'):
    metricas = []
    min_int = [1,2,3,4]
    for m in min_int:
        p, r, f = user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,good_users,x=5,N=N,good_rating=4,
                                min_int=m,u_exp=0,norm_expansion='none',similitud='pearson')
        metricas.append((m,p,r,f))
    plot_interseccion_minima(metricas,N)
elif(sys.argv[2]=='usuarios_experimentados'):
    metricas = []
    user_exp = [0, round(n_items*0.05), round(n_items*0.1), round(n_items*0.15)]
    for u_e in user_exp:
        p, r, f = user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,good_users,x=5,N=N,good_rating=4,
                            min_int=1,u_exp=u_e,norm_expansion='none',similitud='pearson')
        metricas.append((u_e,p,r,f))
    plot_usuarios_experimentados(metricas,N)
elif(sys.argv[2]=='normalizacion_pearson'):
    metricas = []
    norm = ['none','mean_centering_v','mean_centering_u','z_score_v','z_score_u']
    for n_ in norm:
        p, r, f = user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,good_users,x=5,N=N,good_rating=4,
                                min_int=1,u_exp=0,norm_expansion=n_,similitud='pearson')
        metricas.append((n_,p,r,f))
    plot_normalizacion_sup(metricas,N)
elif(sys.argv[2]=='normalizacion_coseno'):
    metricas = []
    norm = ['none','mean_centering_v','mean_centering_u','z_score_v','z_score_u']
    for n_ in norm:
        p, r, f = user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,good_users,x=5,N=N,good_rating=4,
                                min_int=1,u_exp=0,norm_expansion=n_,similitud='coseno')
        metricas.append((n_,p,r,f))
    plot_normalizacion_sup(metricas,N)
elif(sys.argv[2]=='cold_start'):
    metricas = []
    sim = ['coseno', 'pearson']
    given = [2,3,5,10,20]
    for s in sim:
        for g in given:
            p_at_k,r_at_k,f_at_k=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,good_users,x=g,N=100,
                                                good_rating=4,min_int=-1,u_exp=-1,norm_expansion='mean_centering_v',similitud=s)
            metricas.append((s,p_at_k, r_at_k, f_at_k))
    plot_cold_start(metricas,given)