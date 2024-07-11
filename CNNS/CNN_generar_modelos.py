# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:30:49 2022
Generación de modelos de redes convolucionales con variación de tasas de aprendizaje
y guardado de resultados.

@author: JVelez
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import adam_v2
from keras import optimizer_v2
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.utils.random import check_random_state
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from csv import writer
import time



LEARNING_RATE =0.0001
EPOCHS = 1000
CARPETA = 'C:/Users/JVelez/TF_MLP/CNNS/modelos/'
IV_ENTRENAMIENTO=''
IV_VALIDACION=''
IV_EVALUACION=''
IV_PREDICCION=''
MIN = 0
MAX = 0

def cargar_conjunto(conjunto):
    df = pd.read_csv (r'..\datasets\{}.csv'.format(conjunto), sep=';',header=None)
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

def build_model(n_steps,n_features):    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    opt = adam_v2.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer = opt,
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
    model.summary()
    return model

def entrenar_validar(model,x_train,y_train,x_valid,y_valid,x_test,y_test,n_features):   
    #tb = TensorBoard(log_dir=NAME, histogram_freq=1)
    #es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=PACIENCIA)
    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], n_features))
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_valid,y_valid), verbose=0)
   
    print("ETRENAMIENTO TERMINADO")
    print(history.history.keys())
    print("")
    
    eval_score = evaluar_modelo(model, x_test, y_test)
    
    return (history,eval_score)

def evaluar_modelo(model,x_valid,y_valid):
    print("--------------EVALUACION------------------")
    score = model.evaluate(x_valid,y_valid,verbose=1,return_dict=True)
    print('Evaluación '+str(score))
    print("")
    return score

def probar_predicciones(model,x_test,y_test):
    yhat=model.predict(x_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test,yhat))
    mae = tf.keras.metrics.MeanAbsoluteError()
    error_mae = mae(y_test, yhat).numpy()  
    rmspe = (np.sqrt(np.mean(np.square((y_test - yhat) / y_test)))) * 100
    return (rmse,error_mae,rmspe)

def generacion_cubo_predicciones(dataset,nombre_archivo,nuevo_modelo):  
    df = pd.read_csv (r'..\datasets\{}.csv'.format(dataset), sep=';', header=None)
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
    df_result = df.to_numpy()
    df_result = np.append(df_result,Y_pred,axis=1)
    
    np.savetxt("C:/Users/JVelez/TF_MLP/CNNS-EHVH/modelos/cubo_predicciones_{}.csv".format(nombre_archivo),df_result,delimiter=';',fmt='%f')
    
def cargar_modelo(model_folder,model_name):
    nuevo_modelo = keras.models.load_model('C:/Users/JVelez/TF_MLP/CNNS/modelos/{}/{}.h5'.format(model_folder,model_name)) 
    return nuevo_modelo   

def probar_predicciones_MLP(model,dataset):
    df = pd.read_csv (r'..\datasets\{}.csv'.format(dataset), sep=';', header=None)
    df.columns=['FocoX','FocoY','FocoZ','FocoI','PuntoX','PuntoY','PuntoZ','Distancia','EH','EV','PuntoI']
    
    X_new = np.asanyarray(df.drop(columns=['PuntoI']))
    Y_new = np.asanyarray(df[['PuntoI']])
    global IV_PREDICCION 
    IV_PREDICCION = ''+str(np.round(np.min(Y_new),2))+' - '+str(np.round(np.max(Y_new),2))

    scaler_xnew = MinMaxScaler()
    xnew_scale = scaler_xnew.fit_transform(X_new)
    scaler_ynew = MinMaxScaler()
    ynew_scale = scaler_ynew.fit_transform(Y_new)

    Y_pred = model.predict(xnew_scale,verbose=1)
    rmse = tf.keras.metrics.RootMeanSquaredError()
    error_rmse = rmse(ynew_scale, Y_pred).numpy()
    mae = tf.keras.metrics.MeanAbsoluteError()
    error_mae = mae(ynew_scale, Y_pred).numpy()
    
    return (error_rmse,error_mae)    
    modelo = cargar_modelo()
    generacion_cubo_predicciones('dataset_VL_11_PRED','',modelo)
    
def calcular_intervalo_variacion():
    df = pd.read_csv (r'..\datasets\dataset_VL_11_PRED.csv', sep=';', header=None)
    df.columns=['FocoX','FocoY','FocoZ','FocoI','PuntoX','PuntoY','PuntoZ','Distancia','EH','EV','PuntoI']
    print(10**df['PuntoI'].min())
    print(10**df['PuntoI'].max())

def guardar_metricas(model,model_id,history):
    model.save(CARPETA+'metricas_modelo_{}/modelo_{}.h5'.format(model_id,model_id))
    metrics = ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'val_mean_squared_error', 'val_root_mean_squared_error', 'val_mean_absolute_error']    
    for m in metrics:
        data = np.asarray(history.history[m])
        np.savetxt(CARPETA+'metricas_modelo_{}/{}.csv'.format(model_id,m),data,delimiter=',',fmt='%f')

def registrar_reporte(model_id,history,eval_score,predict_score,tiempo):
    ep = len(history.history['mean_squared_error'])    
    data = [model_id,LEARNING_RATE,0,ep,tiempo] #Paciencia es cero
    metrics_entrenamiento = ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error']   
    metrics_validacion = ['val_mean_squared_error', 'val_root_mean_squared_error', 'val_mean_absolute_error']       
    for m in metrics_entrenamiento:
        data = np.append(data,history.history[m][ep-1])
    data=np.append(data,IV_ENTRENAMIENTO)
    
    for m in metrics_validacion:
        data = np.append(data,history.history[m][ep-1])
    data = np.append(data,IV_VALIDACION)
    
    for m in metrics_entrenamiento:
        data = np.append(data,eval_score[m])    
        
    data = np.append(data,IV_EVALUACION)
    data = np.append(data,predict_score)
    data = np.append(data,IV_PREDICCION)
    data = np.append(data,[MIN,MAX])
    with open(CARPETA+'reporte.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(data)
        f_object.close()

            
def generar_errores_escala_real(modelo):
    X,Y=cargar_conjunto("dataset_VL_11")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scale = scaler_x.fit_transform(X)
    y_scale = scaler_y.fit_transform(Y)
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale, y_scale, train_size=0.8, test_size=0.2, random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, train_size=0.5, test_size=0.5, random_state=42)
   
    SETS = ["En","Va","Ev"]
    for C in SETS:
        if C == "En":
            print("ENTRENAMIENTO")
            Y_pred = modelo.predict(x_train,verbose=1)
            rmse = tf.keras.metrics.RootMeanSquaredError()
            print('Predicción MSE {:6f}'.format(np.power(rmse(y_train, Y_pred).numpy(),2)))
            print('Predicción RMSE {:6f}'.format(rmse(y_train, Y_pred).numpy()))
            mae = tf.keras.metrics.MeanAbsoluteError()
            print('Predicción MAE {:6f}'.format(mae(y_train, Y_pred).numpy()))
            
            Y_pred = scaler_y.inverse_transform(Y_pred)
            y_train = scaler_y.inverse_transform(y_train)
            
            Y_pred = np.power(10,Y_pred)
            y_train = np.power(10,y_train)
            
            RMSE = rmse(y_train, Y_pred).numpy()
            MAE = mae(y_train, Y_pred).numpy()
            print('{:6f}'.format(np.power(RMSE,2)))
            print('{:6f}'.format(RMSE))
            print('{:6f}'.format(MAE))
        elif C == "Va":
            print("VALIDACION")
            Y_pred = modelo.predict(x_test,verbose=1)
            rmse = tf.keras.metrics.RootMeanSquaredError()
            print('Predicción MSE {:6f}'.format(np.power(rmse(y_test, Y_pred).numpy(),2)))
            print('Predicción RMSE {:6f}'.format(rmse(y_test, Y_pred).numpy()))
            mae = tf.keras.metrics.MeanAbsoluteError()
            print('Predicción MAE {:6f}'.format(mae(y_test, Y_pred).numpy()))
            
            Y_pred = scaler_y.inverse_transform(Y_pred)
            y_test = scaler_y.inverse_transform(y_test)
            
            Y_pred = np.power(10,Y_pred)
            y_test = np.power(10,y_test)
            
            RMSE = rmse(y_test, Y_pred).numpy()
            MAE = mae(y_test, Y_pred).numpy()
            print('{:6f}'.format(np.power(RMSE,2)))
            print('{:6f}'.format(RMSE))
            print('{:6f}'.format(MAE))           
        elif C == "Ev":   
            print("EVALUACION")
            Y_pred = modelo.predict(x_valid,verbose=1)
            rmse = tf.keras.metrics.RootMeanSquaredError()
            print('Predicción MSE {:6f}'.format(np.power(rmse(y_valid, Y_pred).numpy(),2)))
            print('Predicción RMSE {:6f}'.format(rmse(y_valid, Y_pred).numpy()))
            mae = tf.keras.metrics.MeanAbsoluteError()
            print('Predicción MAE {:6f}'.format(mae(y_valid, Y_pred).numpy()))
            
            Y_pred = scaler_y.inverse_transform(Y_pred)
            y_valid = scaler_y.inverse_transform(y_valid)
            
            Y_pred = np.power(10,Y_pred)
            y_valid = np.power(10,y_valid)
            
            RMSE = rmse(y_valid, Y_pred).numpy()
            MAE = mae(y_valid, Y_pred).numpy()
            print('{:6f}'.format(np.power(RMSE,2)))
            print('{:6f}'.format(RMSE))
            print('{:6f}'.format(MAE))        
    
def run():
    X,Y = cargar_conjunto('dataset_VL_11')   
    
    global MIN 
    MIN = np.min(Y)
    global MAX
    MAX = np.max(Y)
   
    x_scale,y_scale = transformar(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(x_scale, y_scale, train_size=0.8, test_size=0.2, random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, train_size=0.5, test_size=0.5, random_state=42)
    n_features = 1
    n_steps = x_train.shape[1]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))

    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], n_features))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    
    global IV_ENTRENAMIENTO
    IV_ENTRENAMIENTO = ''+str(np.round(np.min(y_train),2))+' - '+str(np.round(np.max(y_train),2))
    global IV_VALIDACION
    IV_VALIDACION = ''+str(np.round(np.min(y_valid),2))+' - '+str(np.round(np.max(y_valid),2))
    global IV_EVALUACION
    IV_EVALUACION = ''+str(np.round(np.min(y_test),2))+' - '+str(np.round(np.max(y_test),2))    
    
    model = build_model(n_steps,n_features)
    history,eval_score = entrenar_validar(model, x_train, y_train, x_valid, y_valid, x_test, y_test, n_features)

    rmse_score,mae_score = probar_predicciones(model,x_test,y_test)    
    return (model, history,eval_score,rmse_score,mae_score) 

 
def generar_modelos():
    i=1    
    LRS = [10,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005,0.000001]
    for LR in LRS:        
        global LEARNING_RATE
        LEARNING_RATE = LR
        inicio = time.time()
        modelo,history,eval_score,rmse_score,mae_score = run()
        tiempo_transcurrido = time.time() - inicio
        guardar_metricas(modelo,i,history)
        predict_score=[rmse_score,mae_score]
        registrar_reporte(i, history, eval_score, predict_score, tiempo_transcurrido)
        i+=1
generar_modelos()

def generar_RMSPE():
    modelo = cargar_modelo('metricas_modelo_9','modelo_9')
    X,Y = cargar_conjunto('dataset_VL_11')   
    
    global MIN 
    MIN = np.min(Y)
    global MAX
    MAX = np.max(Y)
   
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scale = scaler_x.fit_transform(X)
    y_scale = scaler_y.fit_transform(Y)
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale, y_scale, train_size=0.8, test_size=0.2, random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, train_size=0.5, test_size=0.5, random_state=42)
    n_features = 1
    n_steps = x_train.shape[1]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))

    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], n_features))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    
    print("PREDICCION %")
    y_pred = modelo.predict(x_test,verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test)
    
    y_pred = np.power(10,y_pred)
    y_test = np.power(10,y_test)
    rmspe = (np.sqrt(np.mean(np.square((y_test - y_pred) / y_test)))) * 100

    print("rmspe = {}".format(rmspe))
    
#generar_RMSPE()
