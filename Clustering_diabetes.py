#MEMORIA 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
#from sklearn.preprocessing import StandardScaler
from tslearn.clustering import silhouette_score
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [25, 8]
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.impute import SimpleImputer
from tslearn.clustering import KShape

#---
import os
import glob
import pandas as pd
import random
# Usa chdir() para cambiar el directorio
os.chdir(r'C:\Users\Usuario\Desktop\TFM\Python\patient')

extension = 'csv'
todos_los_archivos = [i for i in glob.glob('*.{}'.format(extension))]

# Selecciona aleatoriamente 10 pacientes
pacientes_selec = random.sample(todos_los_archivos, 10)

  
#%%
#Los pacientes seleccionados aleatoriamente son los siguientes: 52,91,10,185,62,218,70,245,290 y 60
archivo= (r'C:\Users\Usuario\Desktop\TFM\Python\patient\Patient_70.csv')
datos= pd.read_csv(archivo, index_col=0)
df52= pd.DataFrame(datos)

# Guardamos en el dataframe las columnas necesarias, con las que vamos a trabajar:
# Fecha, Hora, Carbohidratos, Insulina y Valor de Glucosa
datos_52 = df52[['DeviceDtTmDaysFromEnroll', 'DeviceTm', 'CarbInput', 'InsulinOnBoard', 'GlucoseValue']]
#datos_52.set_index('PtId', inplace=True)

#%% guardar todos los registros de los días que cumplen con la condición (es decir, al menos 3 registros de 'CarbInput' mayores que 0)
#proporcionar un DataFrame que contiene todos los registros de los días que cumplen con la condición para cada paciente. 
# Crear un DataFrame vacío para almacenar todos los registros de los días con suficientes registros
datos_52_carbfiltr = pd.DataFrame(columns=datos_52.columns)

# Iterar sobre cada paciente
for PtId, paciente_data in datos_52.groupby(level=0):
    # Filtrar días con al menos 3 registros de CarbInput mayores que 0
    dias_suficientes52 = paciente_data.groupby('DeviceDtTmDaysFromEnroll').filter(lambda x: (x['CarbInput'] > 0).sum() >= 3)
    
    # Agregar todos los registros de los días que cumplen la condición al DataFrame resultante
    datos_52_carbfiltr = pd.concat([datos_52_carbfiltr, dias_suficientes52])






# Crear un DataFrame vacío para almacenar los datos filtrados
datos_52_carbfiltr_glucfiltr = pd.DataFrame(columns=datos_52_carbfiltr.columns)

# Convertir la columna 'DeviceDtTmDaysFromEnroll' a tipo datetime si aún no lo es
datos_52_carbfiltr['DeviceDtTmDaysFromEnroll'] = pd.to_datetime(datos_52_carbfiltr['DeviceDtTmDaysFromEnroll'], errors='coerce')

# Iterar sobre cada paciente
for PtId, paciente_data52 in datos_52_carbfiltr.groupby(level=0):
    # Calcular la diferencia de tiempo en minutos entre las filas
    paciente_data52['TimeDiff'] = paciente_data52['DeviceDtTmDaysFromEnroll'].diff().dt.total_seconds() / 60
    
    # Identificar valores de glucosa igual a 0, máscara booleana (True o False) que indica dónde los valores de 'GlucoseValue' son igual a 0.
    zero_glucose_mask52 = (paciente_data52['GlucoseValue'] == 0)
    
    # Interpolar valores de glucosa igual a 0 con los valores anteriores y siguientes
    paciente_data52['GlucoseValue'].interpolate(method='linear', inplace=True)
    
    # Filtrar filas con valores de glucosa igual a 0 en un periodo de 30 minutos
    paciente_data_filtrado52 = paciente_data52[~zero_glucose_mask52]
    
    # Agregar los datos filtrados al DataFrame resultante
    datos_52_carbfiltr_glucfiltr = pd.concat([datos_52_carbfiltr_glucfiltr, paciente_data_filtrado52])

# Eliminar la columna 'TimeDiff', no es necesaria
datos_52_carbfiltr_glucfiltr.drop('TimeDiff', axis=1, inplace=True)




#%%

# Convertir la columna 'DeviceDtTmDaysFromEnroll' a tipo datetime
datos_52_carbfiltr_glucfiltr['DeviceDtTmDaysFromEnroll'] = pd.to_datetime(datos_52_carbfiltr_glucfiltr['DeviceDtTmDaysFromEnroll'], errors='coerce')

# Obtener días únicos
days52 = datos_52_carbfiltr_glucfiltr['DeviceDtTmDaysFromEnroll'].dt.date.unique().tolist()

# Obtener el número máximo de secuencias por día
max_seq_length52 = max(datos_52_carbfiltr_glucfiltr.groupby(datos_52_carbfiltr_glucfiltr['DeviceDtTmDaysFromEnroll'].dt.date)['GlucoseValue'].count())

# Crear un conjunto de datos tridimensional inicializado con NaN
trid_datos_52_carbfiltr_glucfiltr = np.full((len(days52), max_seq_length52, 1), np.nan)

# Llenar el conjunto de datos tridimensional con los valores de glucosa
for i, day in enumerate(days52):
    daily_data52 = datos_52_carbfiltr_glucfiltr[datos_52_carbfiltr_glucfiltr['DeviceDtTmDaysFromEnroll'].dt.date == day]['GlucoseValue'].values
    trid_datos_52_carbfiltr_glucfiltr[i, :len(daily_data52), 0] = daily_data52

#%%Este paso se realiza debido a que a partir de la columna 286 hay un numero muy alto de NAN.
# Crear tridim_cort con las primeras 286 columnas
trid_datos_52_carbfiltr_glucfiltr286 = trid_datos_52_carbfiltr_glucfiltr[:, :286, :]


#%%
#Metodo codo y escalado
seed = 0
np.random.seed(seed)
trid_datos_52_carbfiltr_glucfiltr286_scaled = TimeSeriesScalerMeanVariance().fit_transform(trid_datos_52_carbfiltr_glucfiltr286)
sz = trid_datos_52_carbfiltr_glucfiltr286_scaled.shape[1]
for yi in range(12):
    plt.subplot(4, 3, yi + 1)
    plt.plot(trid_datos_52_carbfiltr_glucfiltr286_scaled[yi].ravel(), "k-", alpha=.2)
#     plt.text(0.55, 0.85,'Class Label: %d' % (y_train[yi]))
Sum_of_squared_distances = []
K = range(2,6)
for k in K:
    km = TimeSeriesKMeans(n_clusters=k,
                          n_init=2,
                          metric="dtw",
                          verbose=False,
                          max_iter_barycenter=10,
                          random_state=0)
    
    km = km.fit(trid_datos_52_carbfiltr_glucfiltr286_scaled)
    Sum_of_squared_distances.append(km.inertia_)

# Asegurar de que ambas listas tengan la misma longitud
K = list(K)  # Convierte a lista para asegurar de la longitud
#%%
plt.plot(K[:len(Sum_of_squared_distances)], Sum_of_squared_distances, 'bx-')

plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


#%%
n_clusters = 3
sz = trid_datos_52_carbfiltr_glucfiltr286_scaled.shape[1]
seed = 0

# Aplanar el conjunto de datos tridimensional
flattened_data52 = trid_datos_52_carbfiltr_glucfiltr286_scaled.reshape(trid_datos_52_carbfiltr_glucfiltr286_scaled.shape[0], -1)

# Realizar la imputación para manejar valores NaN
imputer52 = SimpleImputer(strategy='mean')  # Puedes ajustar la estrategia según tus necesidades
flattened_data_imputed52 = imputer52.fit_transform(flattened_data52)

# Aplicar KMeans para obtener las asignaciones de clúster
kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
cluster_assignments = kmeans.fit_predict(flattened_data_imputed52)

# Gráfico de clústeres reales
plt.figure()
for yi in range(n_clusters):
    plt.subplot(1, n_clusters, yi + 1)  # Ajustado a 1 fila para simplicidad
    for xx in flattened_data_imputed52[cluster_assignments == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
        
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85, f'Clúster {yi}', transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Real")

plt.show()

#%%
# Euclidean k-means
print("Euclidean k-means")

km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=False, random_state=seed)
y_pred_km = km.fit_predict(flattened_data_imputed52)
print("Euclidean silhoutte: {:.2f}".format(silhouette_score(flattened_data_imputed52, y_pred_km, metric="euclidean")))

plt.figure()
for yi in range(n_clusters):
    plt.subplot(3, n_clusters, yi + 1)
    for xx in trid_datos_52_carbfiltr_glucfiltr286_scaled[y_pred_km == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")


#%%
# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
                          n_init=2,
                          metric="dtw",
                          verbose=False,
                           max_iter_barycenter=10,
                          random_state=seed)
y_pred_dba_km = dba_km.fit_predict(flattened_data_imputed52)
print("DBA silueta: {:.2f}".format(silhouette_score(flattened_data_imputed52, y_pred_dba_km, metric="dtw")))

for yi in range(n_clusters):
    plt.subplot(3, n_clusters, yi+1)
    for xx in flattened_data_imputed52[y_pred_dba_km == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DBA $k$-means")

plt.tight_layout()
plt.show()

#%%
# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=3, 
                           metric="softdtw",  
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(flattened_data_imputed52)
print("SoftDTW silueta: {:.2f}".format(silhouette_score(flattened_data_imputed52, y_pred, metric='softdtw')))

for yi in range(3):
    plt.subplot(3, 3, 7 + yi)
    for xx in trid_datos_52_carbfiltr_glucfiltr286_scaled[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()


#%%

print("k-shape")
kshape_model = KShape(n_clusters=3,
                     verbose=True,
                     random_state=seed)
y_pred_kshape = kshape_model.fit_predict(flattened_data_imputed52)

for yi in range(3):
    plt.subplot(3, 3, 7 + yi)
    for xx in trid_datos_52_carbfiltr_glucfiltr286_scaled[y_pred_kshape == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(kshape_model.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("K-Shape Clustering")

plt.tight_layout()
plt.show()

# Calcular el coeficiente de silueta
silhouette_avg = silhouette_score(flattened_data_imputed52, y_pred_kshape)

# Imprimir el resultado
print(f"Coeficiente de Silueta para K-Shape: {silhouette_avg}")
#%%MEDIA Y STD POR CLUSTERS



#  dimensiones correctas antes de construir el DataFrame
dim_corr = trid_datos_52_carbfiltr_glucfiltr286.reshape(trid_datos_52_carbfiltr_glucfiltr286.shape[0], -1)

# Crea el DataFrame
df_clustered_MEAN52 = pd.DataFrame(dim_corr, columns=[f'Feature_{i}' for i in range(flattened_data_imputed52.shape[1])])

# Agrega  etiquetas de cluster al DataFrame
df_clustered_MEAN52['Cluster'] = y_pred

# Calcula la media de cada cluster
cluster_means_MEAN52 = df_clustered_MEAN52.groupby('Cluster').mean()
cluster_means_MEANstd52 = df_clustered_MEAN52.groupby('Cluster').std()

# Agrega una columna 'Mean' al DataFrame con el valor medio de todas las columnas
cluster_means_MEAN52['Mean'] = cluster_means_MEAN52.mean(axis=1)
cluster_means_MEANstd52['STD'] = cluster_means_MEANstd52.std(axis=1)


#%% DÍAS TOTALES POR CLUSTER
# Calcula la cantidad de series temporales en cada cluster
num_series_temporales_por_cluster52 = df_clustered_MEAN52['Cluster'].value_counts().sort_index()

# Crea un DataFrame con la información
info_clusters52 = pd.DataFrame({
    'Cluster': num_series_temporales_por_cluster52.index,
    'Dias': num_series_temporales_por_cluster52.values
})

#%% METRICAS
# Crea el DataFrame df_metricas52
df_metricas52 = pd.DataFrame({
    'Cluster': cluster_means_MEAN52.index,
    'Mean': cluster_means_MEAN52['Mean'],
    'STD': cluster_means_MEANstd52['STD'],
    'DIAS_CLUSTER': info_clusters52['Dias']
}, )

# Imprime el nuevo DataFrame df_metricas52
print(df_metricas52)

# Especifica la ruta y el nombre del archivo Excel
ruta_excel521 = r'C:\Users\Usuario\Desktop\TFM\Python\df_metricas52.xlsx'

# Exporta el DataFrame a un archivo Excel
df_metricas52.to_excel(ruta_excel521, index=True)
#%%CREAD DF CON CLUSTERS Y DÍAS

# Crear un nuevo DataFrame con los índices de days
df_clustered_days52 = pd.DataFrame(index=days52, data=df_clustered_MEAN52.values, columns=df_clustered_MEAN52.columns)


#%% CARB E INS DIARIA:

# Calcular el sumatorio de CarbInput e InsulinOnBoard para valores iguales de DeviceDtTmDaysFromEnroll (por días)
sumatoriosCARBINS52 = datos_52.groupby('DeviceDtTmDaysFromEnroll').agg({'CarbInput': 'sum', 'InsulinOnBoard': 'sum'})


# Ordena el DataFrame por el índice
sumatoriosCARBINS52 = sumatoriosCARBINS52.sort_index()

ruta_excel80 = r'C:\Users\Usuario\Desktop\TFM\Python\sumatoriosCARBINS52.xlsx'
ruta_excel81 = r'C:\Users\Usuario\Desktop\TFM\Python\df_clustered_days52.xlsx'
# Exporta el DataFrame a un archivo Excel
sumatoriosCARBINS52.to_excel(ruta_excel80, index=True)
df_clustered_days52.to_excel(ruta_excel81, index=True)

#%% Registros Hipo, Hiper- Tiempo Hipo, hiper (sin tener clusters en cuenta en la contabilización)

import pandas as pd

#  'Hipo' la glucosa es menor a un cierto umbral y 'Hiper' cuando es mayor a otro umbral
umbral_hipo = 70
umbral_hiper = 180

# Filtrar columnas relevantes 
glucosa_columns = [col for col in df_clustered_days52.columns if col.startswith('Feature_')]

# Crear columnas 'Registros Hipo' y 'Registros Hiper' solo en las columnas de glucosa
df_clustered_days52['Registros Hipo'] = (df_clustered_days52[glucosa_columns] < umbral_hipo).sum(axis=1).astype(int)
df_clustered_days52['Registros Hiper'] = (df_clustered_days52[glucosa_columns] > umbral_hiper).sum(axis=1).astype(int)

# Crear columnas 'Tiempo Hipo' y 'Tiempo Hiper'
df_clustered_days52['Tiempo Hipo'] = df_clustered_days52['Registros Hipo'] * 5  # Multiplicar por 5 para obtener minutos
df_clustered_days52['Tiempo Hiper'] = df_clustered_days52['Registros Hiper'] * 5  # Multiplicar por 5 para obtener minutos

# Contabilizar el número total de registros de 'Hipo' y 'Hiper' por día
registros_hipo_por_dia = df_clustered_days52['Registros Hipo'].groupby(level=0).sum()
registros_hiper_por_dia = df_clustered_days52['Registros Hiper'].groupby(level=0).sum()

# Extraer las columnas necesarias al nuevo DataFrame 'registros_hipo_hiper_52'
registros_hipo_hiper_52 = df_clustered_days52[['Cluster', 'Registros Hipo', 'Registros Hiper', 'Tiempo Hipo', 'Tiempo Hiper']].copy()

# Eliminar las columnas 'Registros Hipo', 'Registros Hiper', 'Tiempo Hipo', 'Tiempo Hiper' de df_clustered_days52
df_clustered_days52.drop(['Registros Hipo', 'Registros Hiper', 'Tiempo Hipo', 'Tiempo Hiper'], axis=1, inplace=True)

# Agrupar por 'Cluster' y sumar los valores de Registros Hipo', 'Registros Hiper', 'Tiempo Hipo', 'Tiempo Hiper
sumatorio_cluster_registros_hipo_hiper52 = registros_hipo_hiper_52.groupby('Cluster').sum()


# Especificar la ruta y el nombre del archivo Excel
ruta_excel212 = r'C:\Users\Usuario\Desktop\TFM\Python\sumatorio_cluster_registros_hipo_hiper52.xlsx'

# Exporta el DataFrame a un archivo Excel
sumatorio_cluster_registros_hipo_hiper52.to_excel(ruta_excel212, index=True)


#%%TIR 

import pandas as pd

#  rango de glucosa "en rango"  entre 70 y 180
rango_minimo = 70
rango_maximo = 180

#  DataFrame auxiliar para almacenar el tiempo en rango por cada serie temporal
df_tiempo_en_rango52 = pd.DataFrame(index=df_clustered_days52.index, columns=['TiempoEnRango'])

# Calcular el tiempo en rango para cada serie temporal
for serie_temporal in df_clustered_days52.index:
    tiempo_en_rango52 = df_clustered_days52.loc[serie_temporal].apply(lambda x: 1 if rango_minimo <= x <= rango_maximo else 0).sum() * 5
    df_tiempo_en_rango52.loc[serie_temporal] = tiempo_en_rango52
    df_tiempo_en_rango52.loc[serie_temporal, 'Cluster'] = df_clustered_days52.loc[serie_temporal, 'Cluster']

# Calcular el tiempo total y el porcentaje de tiempo en rango para cada serie temporal
df_tiempo_total = 286 * 5  # Total de minutos en todas las series temporales
df_tiempo_en_rango52['PorcentajeEnRango'] = (df_tiempo_en_rango52['TiempoEnRango'] / df_tiempo_total) * 100

# Agrupar por 'Cluster' y sumar los valores de 'TiempoEnRango' y media de'PorcentajeEnRango'
cluster_registros_tiempoenrango = df_tiempo_en_rango52.groupby('Cluster').agg({
    'TiempoEnRango': 'sum',
    'PorcentajeEnRango': 'mean'
}).reset_index()

# Especificar la ruta y el nombre del archivo Excel
ruta_excel312 = r'C:\Users\Usuario\Desktop\TFM\Python\cluster_registros_tiempoenrango.xlsx'

# Exporta el DataFrame a un archivo Excel
cluster_registros_tiempoenrango.to_excel(ruta_excel312, index=True)

#%%
# Especifica la ruta y el nombre del archivo Excel
ruta_excel52 = r'C:\Users\Usuario\Desktop\TFM\Python\Libro3_4.xlsx'

# Exporta el DataFrame a un archivo Excel
cluster_means_MEAN52.to_excel(ruta_excel52, index=True)



