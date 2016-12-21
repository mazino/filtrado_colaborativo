#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots import *
from funciones import *
from funciones import _good_users
from user_based_comparacion import user_based_CV, _user_based, user_based_ndcg, user_based_entropia,user_based_CV_1M,user_based_ndcg_1M
from time import time
from scipy.sparse import csr_matrix
from scipy.io import mmread

if(sys.argv[2]=='100k'):
    genres = ['unknown', 'Action', 'Adventure', 'Animation','Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
          'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery','Romance','Sci-Fi', 'Thriller', 'War', 'Western']
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
          'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
          'Romance','Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(24), encoding='latin-1')
    hash_v = hash_productos(movies, 'movie_id')
    ratings = csr_matrix(mmread('ml-100k/ratings.mtx'))
elif(sys.argv[2]=='1m' or sys.argv[2]=='1M'):
    genres = ['unknown', 'Action', 'Adventure', 'Animation','Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
          'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery','Romance','Sci-Fi', 'Thriller', 'War', 'Western']
    m_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv('ml-1m/movies.dat', sep='::', names=m_cols, engine='python')
    hash_v = hash_productos(movies, 'movie_id')
    ratings = csr_matrix(mmread('ml-1m/ratings.mtx'))

#m_cols = ['movie_id', 'title', 'genres']
#movies = pd.read_csv('ml-10m/movies.dat', sep='::', names=m_cols, engine='python')
#movies = pd.read_csv('ml-20m/movies.csv', sep=',', names=m_cols)

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

N=int(sys.argv[6])
if(sys.argv[4] == 'alphas'):
    metricas = []
    alphas = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    for a in alphas:
        p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,
            good_users,x=5,N=N,good_rating=4,k_max=40,alpha=a,min_int=1,u_exp=0,norm_expansion='mean_centering_v',
            normalizacion='none',similitud='pearson')
        metricas.append((a,p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib))
    plot_alphas(metricas)
    plot_best_alphas(metricas,N)
elif(sys.argv[4] == 'comparacion_directa'):
    p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,
                    good_users,x=5,N=N,good_rating=4,k_max=40,alpha=0.2,min_int=1,u_exp=0,
                    norm_expansion='mean_centering_v',normalizacion='none',similitud='pearson')
    plot_comparacion_expansiones(p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib,N,hibrido=False)
    plot_comparacion_expansiones(p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib,N,hibrido=True)
elif(sys.argv[4] == 'abundacia_datos'):
    p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib=user_based_CV(m_ratings,n_users,n_items,users_mean,users_std,
                    good_users,x=20,N=N,good_rating=4,k_max=40,alpha=0.2,min_int=1,u_exp=0,
                    norm_expansion='mean_centering_v',normalizacion='none',similitud='pearson')
    plot_comparacion_expansiones(p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib,N,hibrido=False)
    plot_comparacion_expansiones(p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib,N,hibrido=True)
elif(sys.argv[4] == 'comparacion_1M' or sys.argv[2] == 'comparacion_1m'):
    #relevancia binaria para 1M
    print("Calculando metricas de relevancia binaria")
    p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib=user_based_CV_1M(m_ratings,n_users,n_items,users_mean,users_std,
                        good_users,x=5,N=N,good_rating=4,k_max=40,alpha=0.2,min_int=1,u_exp=0,
                        norm_expansion='mean_centering_v',normalizacion='none',similitud='pearson')
    plot_comparacion_expansiones(p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib,N,hibrido=False)
    plot_comparacion_expansiones(p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib,N,hibrido=True)
    #ndcg para 1M
    print("Calculando metricas de relevancia categorica")
    nDCG,nDCG_sup,nDCG_hib=user_based_ndcg_1M(m_ratings,n_users,n_items,users_mean,users_std,good_users,
                            x=5,N=N,good_rating=4,k_max=40,alpha=0.2,min_int=1,u_exp=0,
                            norm_expansion='mean_centering_v',normalizacion='none',similitud='pearson')
    plot_comparacion_expansiones_ndcg(nDCG,nDCG_sup,nDCG_hib,N,hibrido=False)
    plot_comparacion_expansiones_ndcg(nDCG,nDCG_sup,nDCG_hib,N,hibrido=True)
elif(sys.argv[4]=='ndcg'):
    nDCG,nDCG_sup,nDCG_hib=user_based_ndcg(m_ratings,n_users,n_items,users_mean,users_std,good_users,
                            x=5,N=N,good_rating=4,k_max=40,alpha=0.2,min_int=1,u_exp=0,
                            norm_expansion='mean_centering_v',normalizacion='none',similitud='pearson')
    plot_comparacion_expansiones_ndcg(nDCG,nDCG_sup,nDCG_hib,N,hibrido=False)
    plot_comparacion_expansiones_ndcg(nDCG,nDCG_sup,nDCG_hib,N,hibrido=True)
elif(sys.argv[4]=='entropia'):
    e, e_sup, in_, j_ = user_based_entropia(m_ratings,movies,genres,n_users,n_items,users_mean,users_std,
                                    good_users,x=5,N=N,good_rating=4,k_max=40,alpha=0.2,min_int=1,u_exp=0,
                                    norm_expansion='mean_centering_v',normalizacion='none',similitud='pearson')
    plot_entropia(e,e_sup,N)
    print u"\nInteresección para una lista de largo 10: "
    print in_[0]
    print "Jaccard para una lista de largo 10"
    print j_[0]