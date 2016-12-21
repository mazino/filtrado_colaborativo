#!usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#######################Plot expnasion de primer orden#######################
def plot_umbral_relevancia(goodRating, metricas):
    g_rating = np.arange(len(goodRating))

    # Se grafica la precision y recal de acuerdo a la cantidad de ratings relevantes (P@10 y R@10)
    precision = [metricas[0][1][9], metricas[1][1][9],  metricas[2][1][9]]
    recall = [metricas[0][2][9], metricas[1][2][9], metricas[2][2][9]]

    fig, ax = plt.subplots()

    bar_width = 0.35 
    opacity = 0.7
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(g_rating + bar_width, precision, bar_width, alpha=opacity,
                     color='b', error_kw=error_config, label=u'Precisión')

    rects2 = plt.bar(g_rating + 2*bar_width, recall, bar_width, alpha=opacity,
                     color='r', error_kw=error_config, label='Recall')

    plt.xlabel('Ratings', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.title('Umbral de relevancia', fontsize=24)
    ax.set_xticks(g_rating + 2*bar_width)
    ax.set_xticklabels(goodRating)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle = 'solid', alpha= 0.2)

    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., fontsize = 'x-large')
    plt.show()
def plot_varianza(metricas,N):
    top_list = np.linspace(1, N, num=N)

    fig, ax = plt.subplots()
    plt.plot(top_list, metricas[1][1], 'b-o' if N<=30 else 'b-', label=u'Pearson')
    ax.fill_between(top_list, metricas[1][1] + 50*metricas[1][4], metricas[1][1] - 50*metricas[1][4], color='blue', alpha=0.3)
    plt.xlabel('N', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Varianza Inter Folds', fontsize=24, y=1.03)
    plt.grid(True)

    if(N==10):
        plt.xlim(0.6, 10.5)
    plt.ylim(0, 1)

    plt.show()

    fig, ax = plt.subplots()
    plt.plot(top_list, metricas[0][1], 'g-^' if N<=30 else 'g-', label=u'Coseno')
    ax.fill_between(top_list, metricas[0][1] + 50*metricas[0][4], metricas[0][1] - 50*metricas[0][4], color='green', alpha=0.3)
    plt.xlabel('N', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Varianza Inter Folds', fontsize=24, y=1.03)
    plt.grid(True)

    if(N==10):
        plt.xlim(0.6, 10.5)
    plt.ylim(0, 1)
    plt.show()
def plot_tamanno_vecindario(metricas,vecindario_max):
    pearson_f = []
    pearson_p = []
    pearson_r = []
    coseno_f = []
    coseno_p = []
    coseno_r = []
    for i, (name, p, r, f) in enumerate(metricas):
        if(name == 'pearson'):
            pearson_f.append(f[9])
            pearson_p.append(p[9])
            pearson_r.append(r[9])
        elif(name == 'coseno'):
            coseno_f.append(f[9])
            coseno_p.append(p[9])
            coseno_r.append(r[9])
        
    plt.plot(vecindario_max, pearson_f, 'b-o', label= "Pearson")
    plt.plot(vecindario_max, coseno_f, 'g-^', label= "Coseno")
    plt.xlabel(u'Número de vecinos', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    plt.title('kNN en MovieLens 100k', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    plt.xlim(0, 415.0)
    #plt.ylim(0.34, 0.43)
    plt.show()

    plt.plot(vecindario_max, pearson_p, 'b-o', label= "Pearson")
    plt.plot(vecindario_max, coseno_p, 'g-^', label= "Coseno")
    plt.xlabel(u'Número de vecinos', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title('kNN en MovieLens 100k', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    plt.xlim(0, 415.0)
    plt.show()

    plt.plot(vecindario_max, pearson_r, 'b-o', label= "Pearson")
    plt.plot(vecindario_max, coseno_r, 'g-^', label= "Coseno")
    plt.xlabel(u'Número de vecinos', fontsize=20)
    plt.ylabel('Recall', fontsize=20)
    plt.title('kNN en MovieLens 100k', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    plt.xlim(0, 415.0)
    plt.show()
def plot_normalizacion(metricas,N):
    top_list = np.arange(N)
    # Se grafica la precision con similitud de Coseno de acuerdo a las normalizacion y recomendacion top-N
    n_precision = metricas[0][2]
    m_precision = metricas[1][2]
    z_precision = metricas[2][2]

    fig, ax = plt.subplots()

    inicio = 0.6
    bar_width = 0.25

    opacity = 0.7
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(top_list + inicio, n_precision, bar_width, alpha=opacity,
                     color='b', error_kw=error_config, label='none')

    rects2 = plt.bar(top_list + inicio + bar_width, m_precision, bar_width, alpha=opacity,
                     color='r', error_kw=error_config, label='mean_centering')

    rects3 = plt.bar(top_list + inicio + 2*bar_width, z_precision, bar_width, alpha=opacity,
                     color='g', error_kw=error_config, label='z_score')

    plt.xlabel('N',fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Recomendación Top-N: Coseno', fontsize=24)
    plt.xticks(top_list + 1 )
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., fontsize = 'x-large')

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle = 'solid', alpha= 0.2)

    if(N==10):
        plt.xlim(0, 11.0)
    plt.ylim(0, 0.9)

    # Se grafica la precision con similitud de Pearson de acuerdo a las normalizacion y recomendacion top-N
    n_precision = metricas[3][2]
    m_precision = metricas[4][2]
    z_precision = metricas[5][2]

    fig, ax = plt.subplots()

    inicio = 0.6
    bar_width = 0.25

    opacity = 0.7
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(top_list + inicio, n_precision, bar_width, alpha=opacity,
                     color='b', error_kw=error_config, label='none')

    rects2 = plt.bar(top_list + inicio + bar_width, m_precision, bar_width, alpha=opacity,
                     color='r', error_kw=error_config, label='mean_centering')

    rects3 = plt.bar(top_list + inicio + 2*bar_width, z_precision, bar_width, alpha=opacity,
                     color='g', error_kw=error_config, label='z_score')

    plt.xlabel('N',fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Recomendación Top-N: Pearson', fontsize=24)
    plt.xticks(top_list + 1 )
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., fontsize = 'x-large')

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle = 'solid', alpha= 0.2)

    if(N==10):
        plt.xlim(0, 11.0)
    plt.ylim(0, 0.9)
    plt.show()
def plot_productos_recomendados(metricas,N):
    top_list = np.linspace(1, N, num=N)
    for i, (m_name, p, r, f) in enumerate(metricas):
        plt.plot(top_list, p, 'b-o' if N<=30 else 'b-', label= u"Precisión")
        plt.plot(top_list, r, 'g-^' if N<=30 else 'g-', label= "Recall")
        plt.plot(top_list, f, 'r-s' if N<=30 else 'r-', label="F1-score", alpha = 0.8)
        plt.xlabel('N', fontsize=20)
        plt.ylabel('Score', fontsize=20)
        plt.title(u'Recomendación Top-N', fontsize=24)

        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

        if(N==10):
            plt.xlim(0.1, 10.5)
        plt.ylim(0.0, 1)
        plt.show()
def plot_trade_off(metricas,N):
    plt.plot(metricas[0][1], metricas[0][0], 'b-o' if N<=30 else 'b-', alpha = 0.7)
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Precisón/Recall en Recomendación Top-N', fontsize=24, y=1.03)
    plt.grid(True)
    plt.show()
def plot_cold_start(metricas,given):
    precision_pearson = []
    precision_coseno = []
    recall_pearson = []
    recall_coseno = []
    fscore_pearson = []
    fscore_coseno = []
    for i, (s,p,r,f) in enumerate(metricas):
        if(s=='pearson'):
            precision_pearson.append(p[9])
            recall_pearson.append(r[9])
            fscore_pearson.append(f[9])
        elif(s=='coseno'):
            precision_coseno.append(p[9])
            recall_coseno.append(r[9])
            fscore_coseno.append(f[9])

    plt.plot(given, precision_pearson, 'b-o', label= u"Pearson")
    plt.plot(given, precision_coseno, 'g-s', label= u"Coseno")
    plt.xlabel('given', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Escasez de rating', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlim(1.5, 20.5)
    plt.show()

    plt.plot(given, recall_pearson, 'b-o', label= u"Pearson")
    plt.plot(given, recall_coseno, 'g-s', label= u"Coseno")
    plt.xlabel('given', fontsize=20)
    plt.ylabel(u'Recall', fontsize=20)
    plt.title(u'Escasez de rating', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlim(1.5, 20.5)
    plt.show()

    plt.plot(given, fscore_coseno, 'b-o', label= u"Pearson")
    plt.plot(given, fscore_pearson, 'g-s', label= u"Coseno")
    plt.xlabel('given', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    plt.title(u'Escasez de rating', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlim(1.5, 20.5)
    plt.show()
def plot_ndcg(metricas,N):
    top_list = np.linspace(1, N, num=N)
    plt.plot(top_list, metricas[0][1], 'b-o' if N<=30 else 'b-', label = metricas[0][0])
    plt.plot(top_list, metricas[1][1], 'g-^' if N<=30 else 'g-', label = metricas[1][0])

    plt.xlabel('N', fontsize=20)
    plt.ylabel('nDCG', fontsize=20)
    plt.title(u'nDCG en recomendación Top-N', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    #plt.ylim(0.85, 0.95)
    plt.show()

#######################Plot expnasion de segundo orden#######################
def plot_expansion_segundo_orden(metricas,N):
    top_list = np.arange(N) + 1
    for i, (name, p, r, f) in enumerate(metricas):
        if(name == 'pearson'):
            pearson_p=p
            pearson_f=f
            pearson_r=r
        elif(name == 'coseno'):
            coseno_p=p
            coseno_f=f
            coseno_r=r

    plt.plot(top_list, pearson_p, 'b-o' if N<=30 else 'b-', label= "Pearson")
    plt.plot(top_list, coseno_p, 'g-^' if N<=30 else 'g-', label= "Coseno")
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Expansión de segundo orden en MovieLens 100k', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0, 10.5)
    #plt.ylim(0.30, 0.70)
    plt.show()

    plt.plot(top_list, pearson_r, 'b-o' if N<=30 else 'b-', label= "Pearson")
    plt.plot(top_list, coseno_r, 'g-^' if N<=30 else 'g-', label= "Coseno")
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel('Recall', fontsize=20)
    plt.title(u'Expansión de segundo orden en MovieLens 100k', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0, 10.5)
    plt.show()

    plt.plot(top_list, pearson_f, 'b-o' if N<=30 else 'b-', label= "Pearson")
    plt.plot(top_list, coseno_f, 'g-^' if N<=30 else 'g-', label= "Coseno")
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    plt.title(u'Expansión de segundo orden en MovieLens 100k', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0, 10.5)
    plt.show()
def plot_interseccion_minima(metricas,N):
    top_list = np.linspace(1, N, num=N)

    for i, (inter, p, r, f) in enumerate(metricas):
        if(inter == 1):
            name_1 = inter
            inter_1_p = p
            inter_1_r = r
            inter_1_f = f
        elif(inter == 2):
            name_2 = inter
            inter_2_p = p
            inter_2_r = r
            inter_2_f = f
        elif(inter == 3):
            name_3 = inter
            inter_3_p = p
            inter_3_r = r
            inter_3_f = f
        elif(inter == 4):
            name_4 = inter
            inter_4_p = p
            inter_4_r = r
            inter_4_f = f
            
    plt.plot(top_list, inter_1_p, 'b-o' if N<=30 else 'b-', label= "min_intersect=%d"%(name_1))
    plt.plot(top_list, inter_2_p, 'g-^' if N<=30 else 'g-', label= "min_intersect=%d"%(name_2))
    plt.plot(top_list, inter_3_p, 'r-s' if N<=30 else 'r-', label= "min_intersect=%d"%(name_3))
    plt.plot(top_list, inter_4_p, 'm-p' if N<=30 else 'm-', label= "min_intersect=%d"%(name_4))
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Interseción Mínima', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.34, 0.43)

    plt.show()

    plt.plot(top_list, inter_1_r, 'b-o' if N<=30 else 'b-', label= "min_intersect=%d"%(name_1))
    plt.plot(top_list, inter_2_r, 'g-^' if N<=30 else 'g-', label= "min_intersect=%d"%(name_2))
    plt.plot(top_list, inter_3_r, 'r-s' if N<=30 else 'r-', label= "min_intersect=%d"%(name_3))
    plt.plot(top_list, inter_4_r, 'm-p' if N<=30 else 'm-', label= "min_intersect=%d"%(name_4))
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Recall', fontsize=20)
    plt.title(u'Interseción Mínima', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.34, 0.43)

    plt.show()

    plt.plot(top_list, inter_1_f, 'b-o' if N<=30 else 'b-', label= "min_intersect=%d"%(name_1))
    plt.plot(top_list, inter_2_f, 'g-^' if N<=30 else 'g-', label= "min_intersect=%d"%(name_2))
    plt.plot(top_list, inter_3_f, 'r-s' if N<=30 else 'r-', label= "min_intersect=%d"%(name_3))
    plt.plot(top_list, inter_4_f, 'm-p' if N<=30 else 'm-', label= "min_intersect=%d"%(name_4))
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    plt.title(u'Interseción Mínima', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.34, 0.43)

    plt.show()
def plot_usuarios_experimentados(metricas,N):
    top_list = np.linspace(1, N, num=N)
    porcentaje = ['0%','5%', '10%', '15%']

    user_exp_0_p = metricas[0][1]
    user_exp_0_r = metricas[0][2]
    user_exp_0_f = metricas[0][3]

    user_exp_5_p = metricas[1][1]
    user_exp_5_r = metricas[1][2]
    user_exp_5_f = metricas[1][3]

    user_exp_10_p = metricas[2][1]
    user_exp_10_r = metricas[2][2]
    user_exp_10_f = metricas[2][3]

    user_exp_15_p = metricas[3][1]
    user_exp_15_r = metricas[3][2]
    user_exp_15_f = metricas[3][3]

    plt.plot(top_list, user_exp_0_p, 'b-o' if N<=30 else 'b-', label= porcentaje[0])
    plt.plot(top_list, user_exp_5_p, 'g-^' if N<=30 else 'g-', label= porcentaje[1])
    plt.plot(top_list, user_exp_10_p, 'r-s' if N<=30 else 'r-', label= porcentaje[2])
    plt.plot(top_list, user_exp_15_p, 'm-p' if N<=30 else 'm-', label= porcentaje[3])
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Usuarios experimentados', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.55, 0.65)

    plt.show()

    plt.plot(top_list, user_exp_0_r, 'b-o' if N<=30 else 'b-', label= porcentaje[0])
    plt.plot(top_list, user_exp_5_r, 'g-^' if N<=30 else 'g-', label= porcentaje[1])
    plt.plot(top_list, user_exp_10_r, 'r-s' if N<=30 else 'r-', label= porcentaje[2])
    plt.plot(top_list, user_exp_15_r, 'm-p' if N<=30 else 'm-', label= porcentaje[3])
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Recall', fontsize=20)
    plt.title(u'Usuarios experimentados', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.34, 0.43)

    plt.show()

    plt.plot(top_list, user_exp_0_f, 'b-o' if N<=30 else 'b-', label= porcentaje[0])
    plt.plot(top_list, user_exp_5_f, 'g-^' if N<=30 else 'g-', label= porcentaje[1])
    plt.plot(top_list, user_exp_10_f, 'r-s' if N<=30 else 'r-', label= porcentaje[2])
    plt.plot(top_list, user_exp_15_f, 'm-p' if N<=30 else 'm-', label= porcentaje[3])
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    plt.title(u'Usuarios experimentados', fontsize=24)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.34, 0.43)

    plt.show()
def plot_normalizacion_sup(metricas,N):
    top_list = np.linspace(1, N, num=N)

    for i, (norm, p, r, f) in enumerate(metricas):
        if(norm == 'none'):
            name_1=norm
            none_p=p
            none_r=r
            none_f=f
        elif(norm == 'mean_centering_v'):
            name_2=norm
            mean_cen_v_p=p
            mean_cen_v_r=r
            mean_cen_v_f=f
        elif(norm == 'mean_centering_u'):
            name_3=norm
            mean_cen_u_p=p
            mean_cen_u_r=r
            mean_cen_u_f=f
        elif(norm == 'z_score_v'):
            name_4=norm
            z_score_v_p=p
            z_score_v_r=r
            z_score_v_f=f
        elif(norm == 'z_score_u'):
            name_5=norm
            z_score_u_p=p
            z_score_u_r=r
            z_score_u_f=f

    plt.plot(top_list, none_p, 'b-o' if N<=30 else 'b-', label=name_1)
    plt.plot(top_list, mean_cen_v_p, 'g-^' if N<=30 else 'g-', label=name_2)
    plt.plot(top_list, mean_cen_u_p, 'r-s' if N<=30 else 'r-', label=name_3)
    plt.plot(top_list, z_score_v_p, 'm-p' if N<=30 else 'm-', label=name_4)
    plt.plot(top_list, z_score_u_p, 'c-h' if N<=30 else 'c-', label=name_5)

    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Normalización en expansión de segundo orden', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.5, 0.90)

    plt.show()

    plt.plot(top_list, none_r, 'b-o' if N<=30 else 'b-', label=name_1)
    plt.plot(top_list, mean_cen_v_r, 'g-^' if N<=30 else 'g-', label=name_2)
    plt.plot(top_list, mean_cen_u_r, 'r-s' if N<=30 else 'r-', label=name_3)
    plt.plot(top_list, z_score_v_r, 'm-p' if N<=30 else 'm-', label=name_4)
    plt.plot(top_list, z_score_u_r, 'c-h' if N<=30 else 'c-', label=name_5)

    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Recall', fontsize=20)
    plt.title(u'Normalización en expansión de segundo orden', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.5, 0.90)

    plt.show()

    plt.plot(top_list, none_f, 'b-o' if N<=30 else 'b-', label=name_1)
    plt.plot(top_list, mean_cen_v_f, 'g-^' if N<=30 else 'g-', label=name_2)
    plt.plot(top_list, mean_cen_u_f, 'r-s' if N<=30 else 'r-', label=name_3)
    plt.plot(top_list, z_score_v_f, 'm-p' if N<=30 else 'm-', label=name_4)
    plt.plot(top_list, z_score_u_f, 'c-h' if N<=30 else 'c-', label=name_5)

    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    plt.title(u'Normalización en expansión de segundo orden', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.5, 0.90)
    plt.show()
#######################Plot Comparacion entre los metodos#######################
def plot_alphas(metricas):
    alphas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    pearson_f = []
    pearson_p = []
    pearson_r = []

    pearson_p.append(metricas[0][1][9])
    pearson_r.append(metricas[0][2][9])
    pearson_f.append(metricas[0][3][9])

    for i, (a,p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib) in enumerate(metricas):
        pearson_p.append(p_hib[9])
        pearson_r.append(r_hib[9])
        pearson_f.append(f_hib[9])

    pearson_p.append(metricas[5][4][9])
    pearson_r.append(metricas[5][5][9])
    pearson_f.append(metricas[5][6][9])

       
    plt.plot(alphas, pearson_p, 'b-o', label= "Pearson")
    plt.xlabel(r'$ \alpha $', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Método Híbrido', fontsize=24)

    plt.grid(True)

    plt.xlim(-0.05, 1.05)
    #plt.ylim(0.80, 0.85)

    plt.show()

    plt.plot(alphas, pearson_r, 'b-o', label= "Pearson")
    plt.xlabel(r'$ \alpha $', fontsize=20)
    plt.ylabel(u'Recall', fontsize=20)
    plt.title(u'Método Híbrido', fontsize=24)

    plt.grid(True)

    plt.xlim(-0.05, 1.05)
    #plt.ylim(0.80, 0.85)

    plt.show()

    plt.plot(alphas, pearson_f, 'b-o', label= "Pearson")
    plt.xlabel(r'$ \alpha $', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    plt.title(u'Método Híbrido', fontsize=24)

    plt.grid(True)

    plt.xlim(-0.05, 1.05)
    #plt.ylim(0.80, 0.85)

    plt.show()
def plot_best_alphas(metricas,N):
    lista = np.linspace(1, N, num=N)

    pearson_p_0=metricas[0][1]
    pearson_r_0=metricas[0][2]
    pearson_f_0=metricas[0][3]

    pearson_p_1=metricas[0][7]
    pearson_r_1=metricas[0][8]
    pearson_f_1=metricas[0][9]

    pearson_p_2=metricas[1][7]
    pearson_r_2=metricas[1][8]
    pearson_f_2=metricas[1][9]

    pearson_p_3=metricas[1][4]
    pearson_r_3=metricas[1][5]
    pearson_f_3=metricas[1][6]
       
    plt.plot(lista, pearson_p_0, 'b-o' if N<=30 else 'b-', label= r'$ \alpha $ = 0,0')
    plt.plot(lista, pearson_p_1, 'g-^' if N<=30 else 'g-', label= r'$ \alpha $ = 0,1')
    plt.plot(lista, pearson_p_2, 'r-s' if N<=30 else 'r-', label= r'$ \alpha $ = 0,2')
    plt.plot(lista, pearson_p_3, 'm-p' if N<=30 else 'm-', label= r'$ \alpha $ = 1,0')
    plt.xlabel('N', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    plt.title(u'Método Híbrido', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.80, 0.86)

    plt.show()

    plt.plot(lista, pearson_r_0, 'b-o' if N<=30 else 'b-', label= r'$ \alpha $ = 0,0')
    plt.plot(lista, pearson_r_1, 'g-^' if N<=30 else 'g-', label= r'$ \alpha $ = 0,1')
    plt.plot(lista, pearson_r_2, 'r-s' if N<=30 else 'r-', label= r'$ \alpha $ = 0,2')
    plt.plot(lista, pearson_r_3, 'm-p' if N<=30 else 'm-', label= r'$ \alpha $ = 1,0')
    plt.xlabel('N', fontsize=20)
    plt.ylabel(u'Recall', fontsize=20)
    plt.title(u'Método Híbrido', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.80, 0.86)

    plt.show()

    plt.plot(lista, pearson_f_0, 'b-o' if N<=30 else 'b-', label= r'$ \alpha $ = 0,0')
    plt.plot(lista, pearson_f_1, 'g-^' if N<=30 else 'g-', label= r'$ \alpha $ = 0,1')
    plt.plot(lista, pearson_f_2, 'r-s' if N<=30 else 'r-', label= r'$ \alpha $ = 0,2')
    plt.plot(lista, pearson_f_3, 'm-p' if N<=30 else 'm-', label= r'$ \alpha $ = 1,0')
    plt.xlabel('N', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    plt.title(u'Método Híbrido', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.80, 0.86)
    plt.show()
def plot_comparacion_expansiones(p,r,f,p_sup,r_sup,f_sup,p_hib,r_hib,f_hib,N,hibrido=False):
    lista = np.linspace(1, N, num=N)
    plt.plot(lista, p, 'b-o' if N<=30 else 'b-', label= u"Exp. 1° orden")
    plt.plot(lista, p_sup, 'g-^' if N<=30 else 'g-', label= u"Exp. 2° orden")
    if(hibrido):
        plt.plot(lista, p_hib, 'r-s' if N<=30 else 'r-', label= u"Híbrido, "+ r"$ \alpha $ = 0,2")
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Precisión', fontsize=20)
    if(hibrido==False):
        plt.title(u'Expansión de primer y segundo orden', fontsize=24, y=1.03)
    else:
        plt.title(u'Expansiones y método Híbrido', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.80, 0.85)
    plt.show()

    plt.plot(lista, r, 'b-o' if N<=30 else 'b-', label= u"Exp. 1° orden")
    plt.plot(lista, r_sup, 'g-^' if N<=30 else 'g-', label= u"Exp. 2° orden")
    if(hibrido):
        plt.plot(lista, r_hib, 'r-s' if N<=30 else 'r-', label= u"Híbrido, "+ r"$ \alpha $ = 0,2")
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Recall', fontsize=20)
    if(hibrido==False):
        plt.title(u'Expansión de primer y segundo orden', fontsize=24, y=1.03)
    else:
        plt.title(u'Expansiones y método Híbrido', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.80, 0.85)
    plt.show()

    plt.plot(lista, f, 'b-o' if N<=30 else 'b-', label= u"Exp. 1° orden")
    plt.plot(lista, f_sup, 'g-^' if N<=30 else 'g-', label= u"Exp. 2° orden")
    if(hibrido):
        plt.plot(lista, f_hib, 'r-s' if N<=30 else 'r-', label= u"Híbrido, "+ r"$ \alpha $ = 0,2")
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'F-score', fontsize=20)
    if(hibrido==False):
        plt.title(u'Expansión de primer y segundo orden', fontsize=24, y=1.03)
    else:
        plt.title(u'Expansiones y método Híbrido', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.80, 0.85)
    plt.show()
def plot_comparacion_expansiones_ndcg(nDCG,nDCG_sup,nDCG_hib,N,hibrido=False):
    top_list = np.linspace(1, N, num=N)

    plt.plot(top_list, nDCG, 'b-o' if N<=30 else 'b-', label=u"Exp. 1° orden")
    plt.plot(top_list, nDCG_sup, 'g-^' if N<=30 else 'g-', label=u"Exp. 2° orden")
    if(hibrido):
        plt.plot(top_list, nDCG_hib, 'r-s' if N<=30 else 'r-', label= u"Híbrido, "+ r"$ \alpha $ = 0,2")

    plt.xlabel('N', fontsize=20)
    plt.ylabel('nDCG', fontsize=20)
    if(hibrido==False):
        plt.title(u'nDCG para expansión de primer y segundo orden', fontsize=24, y=1.03)
    else:
        plt.title(u'nDCG para Expansiones y método Híbrido', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
    #plt.ylim(0.85, 0.95)
    plt.show()
def plot_entropia(e,e_sup,N):
    lista = np.linspace(1,N,N)

    plt.plot(lista, e, 'b-o' if N<=30 else 'b-', label= u"Exp. 1° orden")
    plt.plot(lista, e_sup, 'g-^' if N<=30 else 'g-', label= u"Exp. 2° orden")
    plt.xlabel(u'N', fontsize=20)
    plt.ylabel(u'Entropía', fontsize=20)
    plt.title(u'Entropía en expansión de primer y segundo orden', fontsize=24, y=1.03)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if(N==10):
        plt.xlim(0.5, 10.5)
        plt.ylim(0.72, 1.0)
    plt.show()