#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funciones import *
from funciones import user_mean_std
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
#elif(sys.argv[2]=='10m' or sys.argv[2]=='10M'):
    #genres = ['unknown', 'Action', 'Adventure', 'Animation','Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
    #      'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery','Romance','Sci-Fi', 'Thriller', 'War', 'Western']
    #m_cols = ['movie_id', 'title', 'genres']
    #movies = pd.read_csv('ml-10m/movies.dat', sep='::', names=m_cols, engine='python')
    #hash_v = hash_productos(movies, 'movie_id')
    #ratings = csr_matrix(mmread('ml-10m/ratings.mtx'))
#elif(sys.argv[2]=='20m' or sys.argv[2]=='20M'):
    #genres = ['unknown', 'Action', 'Adventure', 'Animation','Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
    #      'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery','Romance','Sci-Fi', 'Thriller', 'War', 'Western']
    #m_cols = ['movie_id', 'title', 'genres']
    #movies = pd.read_csv('ml-20m/movies.csv', sep=',', names=m_cols)
    #hash_v = hash_productos(movies, 'movie_id')
    #ratings = csr_matrix(mmread('ml-20m/ratings.mtx'))

# Filtrado de productos que no poseen calificaciones (en caso de existir)
m_ratings = ratings[:,hash_v.values()]
# Item based: Transponer la matriz de ratings
#m_ratings = m_ratings.T
n_users = m_ratings.shape[0]
n_items = m_ratings.shape[1]
print(u'Número de usuarios = {} | Número de productos = {}'.format(n_users, n_items))

if(sys.argv[4]=='dispersion'):
    fig, ax = plt.subplots(figsize=(7,7))
    plt.title(('MovieLens %s'%sys.argv[2]), fontsize=24, y=1.03)
    im = ax.imshow(m_ratings.todense() if sys.argv[2]=='100k' else m_ratings.todense().T,
                    origin='lower',cmap='YlOrRd')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)
    ax.xaxis.set_label_text('Productos', fontsize=20)
    ax.yaxis.set_label_text('Usuarios', fontsize=20)
    plt.show()
    print "Dispersion: {:.5f}".format(1 - float(m_ratings.nnz)/(n_users*n_items))
    print "Dimnesion: {:} x {:}".format(n_users, n_items)
elif(sys.argv[4]=='frecuencia_rating'):
    frecuencia_relativa=[]
    rating_frec={1.0:0.0,2.0:0.0,3.0:0.0,4.0:0.0,5.0:0.0}
    ratings_no_nulos = m_ratings.nnz
    for u in range(0, n_users):
        for v in m_ratings[u].nonzero()[1]:
            rating_frec[m_ratings[u,v]]+=1
    for key, value in rating_frec.items():
        frecuencia_relativa.append(value/ratings_no_nulos)
    n_groups = 5
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    inicio = 0.75
    bar_width = 0.5
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index+inicio,frecuencia_relativa,bar_width,alpha=opacity,
                     color='b',error_kw=error_config,label='none')
    plt.xlabel('Rating', fontsize=20)
    plt.ylabel('Frecuencia relativa', fontsize=20)
    plt.title(('MovieLens %s'%sys.argv[2]), fontsize=24, y=1.03)
    plt.xticks(index + 1, ('1', '2', '3', '4', '5'))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle = 'solid', alpha= 0.2)
    plt.xlim(0.5, 5.5)
    plt.tight_layout()
    plt.show()
elif(sys.argv[4]=='distribucion'):
    productos = []
    for v in range(0, n_items):
        productos.append(float(m_ratings[:,v].nnz))
    productos.sort()
    productos = productos[::-1]
    items_percent = np.linspace(1, n_items, num=n_items, dtype=float)*100/n_items
    fig, ax = plt.subplots()
    plt.plot(items_percent, productos, 'o', alpha = 0.5)
    plt.xlabel('% de productos', fontsize=20)
    plt.ylabel('Cantidad de ratings', fontsize=20)
    plt.title(u'Distribución en MovieLens %s'%sys.argv[2], fontsize=24, y=1.03)
    plt.grid(True, linestyle = 'solid', alpha= 0.2)
    plt.show()
elif(sys.argv[4]=='clasificacion_usuarios'):
    users_mean, users_std = user_mean_std(m_ratings, n_users)
    percentil = np.percentile(users_mean, [33,66])
    pesimista = []
    neutro = []
    optimista = []
    for u in range(0, n_users):
        if(users_mean[u] <= percentil[0]):
            pesimista.append(users_mean[u])
        elif(users_mean[u] > percentil[0] and  users_mean[u] <= percentil[1]):
            neutro.append(users_mean[u])
        elif(users_mean[u] > percentil[1]):
            optimista.append(users_mean[u])
    data_to_plot = [pesimista, neutro, optimista]
    labels = ["Pesimista", "Neutro", "Optimista"]
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"], fontsize=20)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle = 'solid', alpha= 0.2)
    plt.show()
elif(sys.argv[4]=='distribucion_generos'):
    if(sys.argv[2]=='100k'):
        clases= {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0,
                10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0}
        for v in range(0, n_items):
            g = movies.loc[v, genres].tolist()
            for i in range(0, len(g)):
                clases[i]+=g[i]
    elif(sys.argv[2]=='1m' or sys.argv[2]=='1M'):
        clases= {'unknown':0,'Action':0,'Adventure':0,'Animation':0,"Children's":0,'Comedy':0,'Crime':0,
                'Documentary':0,'Drama':0,'Fantasy':0,'Film-Noir':0,'Horror':0,'Musical':0,'Mystery':0,
                'Romance':0,'Sci-Fi':0,'Thriller':0,'War':0,'Western':0}
        for v in range(0, n_items):
            g = movies.loc[v,'genres'].split('|')
            for genero in g:
                clases[genero]+=1
    cantidad_generos = sorted(clases.items(), key=lambda x: x[1], reverse=True)
    generos, cantidad= zip(*cantidad_generos)
    generos = list(generos)
    cantidad = list(cantidad)
    if(sys.argv[2]=='100k'):
        g=[]
        for i in generos:
            g.append(genres[i])
    fig, ax = plt.subplots()
    inicio = 0.75
    bar_width = 0.5
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(range(0, len(generos)),cantidad,bar_width,alpha=opacity,
                    tick_label=g if sys.argv[2]=='100k' else generos,color='b',error_kw=error_config,label='none')
    plt.xlabel(u'Géneros', fontsize=20)
    plt.ylabel(u'Frecuencia Absoluta', fontsize=20)
    plt.title(u'Distribución de Géneros', fontsize=24, y=1.03)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45, size=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle = 'solid', alpha= 0.2)
    plt.show()