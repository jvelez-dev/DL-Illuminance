# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 08:46:37 2022
PRUEBA CON EL MUESTREO UNIFORME
@author: JVelez
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import savetxt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras import layers,models


tf.keras.backend.clear_session()#restaura el estado del portátil
NAME = "logs/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

NEURONAS = 30
EPOCHS = 1000
PACIENCIA = 10
CARPETA = './modelos/Version4/3D{}N/'.format(PACIENCIA)
#####################################
#CALLBACKS
#####################################
tb = TensorBoard(log_dir=NAME, histogram_freq=1)
es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=PACIENCIA)


def cargar_conjunto(conjunto):
    df = pd.read_csv (r'.\datasets\{}.csv'.format(conjunto), sep=',',header=None)
    df.columns=['FocoX','FocoY','FocoZ','FocoI','PuntoX','PuntoY','PuntoZ','Distancia','EH','EV','PuntoI']
    X = np.asanyarray(df.drop(columns=['PuntoI']))
    Y = np.asanyarray(df[['PuntoI']])
    return (X,Y)

def transformar(X,Y):
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scale = scaler_x.fit_transform(X)
    y_scale = scaler_y.fit_transform(Y)
    return (x_scale,y_scale)

def build_model():
    #####################################
    #DEFINICION DEL MODELO (ARQUITECTURA)
    #####################################
    model = models.Sequential()
    model.add(layers.Dense(units=NEURONAS,activation='relu',input_dim=10))
    model.add(layers.Dense(units=1,activation='linear'))
    # #cargar checkpoint
    # #model.load_weights("weights.best.hdf5")
    
    model.compile(optimizer= keras.optimizers.Adam(learning_rate=0.01), 
                  loss='mean_squared_error',  
                  metrics=['mse',tf.keras.metrics.RootMeanSquaredError(),'mae'])
    model.summary()
    return model

# #######################################
# #PRUEBA DE PREDICCION DE NUEVOS VALORES
# #######################################    
def generacion_cubo_predicciones(dataset,nombre_archivo,nuevo_modelo):  
    df = pd.read_csv (r'.\datasets\{}.csv'.format(dataset), sep=';', header=None)
    df.columns=['FocoX','FocoY','FocoZ','FocoI','PuntoX','PuntoY','PuntoZ','Distancia','EH','EV','PuntoI']
    
    X_new = np.asanyarray(df.drop(columns=['PuntoI']))
    Y_new = np.asanyarray(df[['PuntoI']])
    scaler_xnew = MinMaxScaler()
    xnew_scale = scaler_xnew.fit_transform(X_new)
    scaler_ynew = MinMaxScaler()
    ynew_scale = scaler_ynew.fit_transform(Y_new)

    Y_pred = nuevo_modelo.predict(xnew_scale,verbose=1)
    rmse = tf.keras.metrics.RootMeanSquaredError()
    print('Predicción RMSE '+str(nombre_archivo)+str(rmse(ynew_scale, Y_pred).numpy()))
    mae = tf.keras.metrics.MeanAbsoluteError()
    print('Predicción MAE '+str(nombre_archivo)+str(mae(ynew_scale, Y_pred).numpy()))
    
    Y_pred = scaler_ynew.inverse_transform(Y_pred)
    #### REVISAR LA GENERACION DEL CUBO PORQUE LES ESTA PONIENDO UN 1 A LOS ESPERADOS QUE COINCIDEN CERCA A LA FUENTE
    df_result = df.to_numpy()
    df_result = np.append(df_result,Y_pred,axis=1)
    
    #savetxt("C:/Users/JVelez/TF_MLP/modelos/Aprox3-earlystopping/Paciencia {}/3D{}{}N/cubo_predicciones_{}.csv".format(PACIENCIA,NEURONAS,nombre_archivo),df_result,delimiter=';',fmt='%f')
    
def entrenar_validar(model,x_train,y_train,x_valid,y_valid,x_eval,y_eval):   
    #####################################
    #ENTRENAMIENTO Y VALIDACIÓN
    #####################################
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_valid,y_valid),verbose=0,callbacks=[es,tb])
    model.save(CARPETA+'modelo_3D_{}N.h5'.format(NEURONAS))
    
    data = asarray(history.history['mse'])
    savetxt(CARPETA+'mse_3D_{}N.csv'.format(NEURONAS),data,delimiter=',',fmt='%f')
    
    data = asarray(history.history['val_mse'])
    savetxt(CARPETA+'val_mse_3D_{}N.csv'.format(NEURONAS),data,delimiter=',',fmt='%f')
    
    data = asarray(history.history['mae'])
    savetxt(CARPETA+'mae_3D_{}N.csv'.format(NEURONAS),data,delimiter=',',fmt='%f')
    
    data = asarray(history.history['val_mae'])
    savetxt(CARPETA+'val_mae_3D_{}N.csv'.format(NEURONAS),data,delimiter=',',fmt='%f')
    
    data = asarray(history.history['root_mean_squared_error'])
    savetxt(CARPETA+'rmse_3D_{}N.csv'.format(NEURONAS),data,delimiter=',',fmt='%f')
    
    data = asarray(history.history['val_root_mean_squared_error'])
    savetxt(CARPETA+'val_rmse_3D_{}N.csv'.format(NEURONAS),data,delimiter=',',fmt='%f')
    
    print("ETRENAMIENTO TERMINADO")
    print(history.history.keys())
    print("")
    print("--------------EVALUACION------------------")
    score = model.evaluate(x_eval,y_eval,verbose=1,return_dict=True)
    print('Evaluación '+str(score))
    print("")
    
    ## PREDICCION GENERACION DEL NUEVO CUBO ##
    #generacion_cubo_predicciones('dataset_3D_{}'.format(VALORES),'ES_{}'.format(VALORES),nuevo_modelo=model)
    
    # y_pred = model.predict(x_test)
    # rmse = tf.keras.metrics.RootMeanSquaredError()
    # print('Predicción RMSE '+str(rmse(y_test, y_pred).numpy()))


def cargar_modelo():
    nuevo_modelo = keras.models.load_model('C:/Users/JVelez/TF_MLP/modelos/Version4/3D10N/modelo_3D_30N.h5') 
    return nuevo_modelo   

def juntar(A,B,C):
    A=np.append(A,B,axis=0)
    A=np.append(A,C,axis=0)
    return A

def particionar(A):
    train = A[0:864]
    valid = A[864:(864+432)]
    eva = A[-432:]
    return (train,valid,eva)

def construir_entrenar_evaluar():    
    x_train,y_train = cargar_conjunto("Entrenamiento_uniforme")
    x_valid,y_valid = cargar_conjunto("Validacion_uniforme")
    x_eval,y_eval = cargar_conjunto("Evaluacion_uniforme")
    
    X = juntar(x_train,x_valid,x_eval)
    Y = juntar(y_train,y_valid,y_eval)
    
    x_scale,y_scale = transformar(X, Y)
    
    x_train,x_valid,x_eval=particionar(x_scale)
    y_train,y_valid,y_eval=particionar(y_scale)
    
    model = build_model()
    
    entrenar_validar(model, x_train, y_train, x_valid, y_valid, x_eval, y_eval)

def probar_predicciones():
    modelo = cargar_modelo()
    generacion_cubo_predicciones('dataset_VL_11_PRED','',modelo)
    
def cakcular_intervalo_variacion():
    df = pd.read_csv (r'.\datasets\dataset_VL_11_PRED.csv', sep=';', header=None)
    df.columns=['FocoX','FocoY','FocoZ','FocoI','PuntoX','PuntoY','PuntoZ','Distancia','EH','EV','PuntoI']
    print(10**df['PuntoI'].min())
    print(10**df['PuntoI'].max())
    

    