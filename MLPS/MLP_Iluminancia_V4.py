# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:38:19 2022

Cambiar neuronas a 10 y 90

TOMAR EL MODELO 52{
    * Escala logarítmica
    * 30 Neuronas
    * 1 capa oculta
    * ADAM
    * tasa 0.0001
    * Early stopping paciencia 80
    * Métricas MSE RMSE MAE
    * Muestreo aleatorio 50-25-25
    * Keras tunning:
        *tasa
        *paciencia
        *neuronas
        *
    }
@author: JVelez
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import time
from csv import writer



hist_df = pd.DataFrame()

tf.keras.backend.clear_session()#restaura el estado del portátil
NAME = "logs/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

NEURONAS = 90
LEARNING_RATE =0.0001
EPOCHS = 5000
PACIENCIA = 80
CARPETA = 'C:/Users/JVelez/TF_MLP/modelos/Version5/'
IV_ENTRENAMIENTO=''
IV_VALIDACION=''
IV_EVALUACION=''
IV_PREDICCION=''
MIN = 0
MAX = 0
#####################################
#CALLBACKS
#####################################



def cargar_conjunto(conjunto):
    df = pd.read_csv (r'.\datasets\{}.csv'.format(conjunto), sep=';',header=None)
    df.columns=['FocoX','FocoY','FocoZ','FocoI','PuntoX','PuntoY','PuntoZ','Distancia','EH','EV','PuntoI']
    df = df.drop(columns=['EH','EV'])#
    X = np.asanyarray(df.drop(columns=['PuntoI']))
    Y = np.asanyarray(df[['PuntoI']])
    return (X,Y)

def transformar(X,Y):
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scale = scaler_x.fit_transform(X)
    y_scale = scaler_y.fit_transform(Y)
    return (x_scale,y_scale)

def build_model(N,LR):
    model = models.Sequential()
    model.add(layers.Dense(units=N,activation='relu',input_dim=10))
    model.add(layers.Dense(units=1,activation='linear'))
    # #cargar checkpoint
    # #model.load_weights("weights.best.hdf5")
    
    model.compile(optimizer= keras.optimizers.Adam(learning_rate=LR), 
                  loss='mean_squared_error',  
                  metrics=['mse',tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
    model.summary()
    return model

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
    df_result = df.to_numpy()
    df_result = np.append(df_result,Y_pred,axis=1)
    
    np.savetxt("C:/Users/JVelez/TF_MLP/modelos/Version4/cubo_predicciones_{}.csv".format(nombre_archivo),df_result,delimiter=';',fmt='%f')
    
def entrenar_validar(model,x_train,y_train,x_valid,y_valid,x_eval,y_eval):   
    tb = TensorBoard(log_dir=NAME, histogram_freq=1)
    es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=PACIENCIA)
    
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_valid,y_valid),verbose=0,callbacks=[es,tb])
   
    print("ETRENAMIENTO TERMINADO")
    print(history.history.keys())
    print("")
    
    eval_score = evaluar_modelo(model, x_eval, y_eval)
    
    return (history,eval_score)

def evaluar_modelo(model,x_valid,y_valid):
    print("--------------EVALUACION------------------")
    score = model.evaluate(x_valid,y_valid,verbose=1,return_dict=True)
    print('Evaluación '+str(score))
    print("")
    return score
    
def cargar_modelo(model_name):
    nuevo_modelo = keras.models.load_model('C:/Users/JVelez/TF_MLP/modelos/Version5/{}/{}.h5'.format(model_name,model_name)) 
    return nuevo_modelo   

def probar_predicciones(model,dataset):
    df = pd.read_csv (r'.\datasets\{}.csv'.format(dataset), sep=';', header=None)
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
    # modelo = cargar_modelo()
    # generacion_cubo_predicciones('dataset_VL_11_PRED','',modelo)
    
def calcular_intervalo_variacion():
    df = pd.read_csv (r'.\datasets\dataset_VL_11_PRED.csv', sep=';', header=None)
    df.columns=['FocoX','FocoY','FocoZ','FocoI','PuntoX','PuntoY','PuntoZ','Distancia','EH','EV','PuntoI']
    print(10**df['PuntoI'].min())
    print(10**df['PuntoI'].max())

def construir_entrenar_validar():    
   X,Y=cargar_conjunto("dataset_VL_11")
   global MIN 
   MIN = np.min(Y)
   global MAX
   MAX = np.max(Y)
   x_scale,y_scale = transformar(X, Y)
   x_train, x_test, y_train, y_test = train_test_split(x_scale, y_scale, train_size=0.5, test_size=0.5, random_state=42)
   x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, train_size=0.5, test_size=0.5, random_state=42)
   
   global IV_ENTRENAMIENTO
   IV_ENTRENAMIENTO = ''+str(np.round(np.min(y_train),2))+' - '+str(np.round(np.max(y_train),2))
   global IV_VALIDACION
   IV_VALIDACION = ''+str(np.round(np.min(y_test),2))+' - '+str(np.round(np.max(y_test),2))
   global IV_EVALUACION
   IV_EVALUACION = ''+str(np.round(np.min(y_valid),2))+' - '+str(np.round(np.max(y_valid),2))

   model = build_model(NEURONAS,LEARNING_RATE)
    
   history,score = entrenar_validar(model, x_train, y_train, x_test, y_test, x_valid, y_valid)
     
   return (model, history,score)
       

def graficar(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
def guardar_metricas(model,model_id,history):
    model.save(CARPETA+'modelo_{}/modelo_{}.h5'.format(model_id,model_id))
    metrics = ['mse', 'root_mean_squared_error', 'mean_absolute_error', 'val_mse', 'val_root_mean_squared_error', 'val_mean_absolute_error']    
    for m in metrics:
        data = np.asarray(history.history[m])
        np.savetxt(CARPETA+'modelo_{}/{}.csv'.format(model_id,m),data,delimiter=',',fmt='%f')

def registrar_reporte(model_id,history,eval_score,predict_score,tiempo):
    ep = len(history.history['mse'])    
    data = [model_id,LEARNING_RATE,PACIENCIA,ep,tiempo]
    metrics_entrenamiento = ['mse', 'root_mean_squared_error', 'mean_absolute_error']   
    metrics_validacion = ['val_mse', 'val_root_mean_squared_error', 'val_mean_absolute_error']       
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
    with open('./modelos/Version5/reporte.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(data)
        f_object.close()

#71 es el numero de modelo que sigue despues de los de la version 4. 71 para 10 neuronas y 72 para 90 neuronas
def generar_modelo(i):
    inicio = time.time()
    modelo,history,eval_score = construir_entrenar_validar()
    tiempo_transcurrido = time.time() - inicio
    error_rmse,error_mae=probar_predicciones(modelo, 'dataset_VL_11_PRED')
    predict_score = [error_rmse,error_mae]
    guardar_metricas(modelo,i,history)
    registrar_reporte(i, history, eval_score, predict_score, tiempo_transcurrido)


def generar_errores_escala_real(modelo):
    X,Y=cargar_conjunto("dataset_VL_11")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scale = scaler_x.fit_transform(X)
    y_scale = scaler_y.fit_transform(Y)
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale, y_scale, train_size=0.5, test_size=0.5, random_state=42)
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
        
def generar_errores_prediccion_escala_real(model,dataset):
    df = pd.read_csv (r'.\datasets\{}.csv'.format(dataset), sep=';', header=None)
    df.columns=['FocoX','FocoY','FocoZ','FocoI','PuntoX','PuntoY','PuntoZ','Distancia','EH','EV','PuntoI']
    
    X_new = np.asanyarray(df.drop(columns=['PuntoI']))
    Y_new = np.asanyarray(df[['PuntoI']])

    scaler_xnew = MinMaxScaler()
    xnew_scale = scaler_xnew.fit_transform(X_new)
    scaler_ynew = MinMaxScaler()
    ynew_scale = scaler_ynew.fit_transform(Y_new)

    Y_pred = model.predict(xnew_scale,verbose=1)
    rmse = tf.keras.metrics.RootMeanSquaredError()
    print('Predicción MSE {:6f}'.format(np.power(rmse(ynew_scale, Y_pred).numpy(),2)))
    print('Predicción RMSE {:6f}'.format(rmse(ynew_scale, Y_pred).numpy()))    
    mae = tf.keras.metrics.MeanAbsoluteError()
    print('Predicción MAE {:6f}'.format(mae(ynew_scale, Y_pred).numpy()))
    
    Y_pred = scaler_ynew.inverse_transform(Y_pred)
    ynew_scale = scaler_ynew.inverse_transform(ynew_scale)
    
    Y_pred = np.power(10,Y_pred)
    ynew_scale = np.power(10,ynew_scale)
    
    RMSE = rmse(ynew_scale, Y_pred).numpy()
    MAE = mae(ynew_scale, Y_pred).numpy()
    print('{:6f}'.format(np.power(RMSE,2)))
    print('{:6f}'.format(RMSE))
    print('{:6f}'.format(MAE))

    
# generar_modelo(72)
#modelo = cargar_modelo("modelo_72")
#generar_errores_prediccion_escala_real(modelo,"dataset_432_VL_PRED")
# generar_errores_escala_real(modelo)
# generacion_cubo_predicciones("dataset_432_VL_PRED", "modelo_52", modelo)


def generar_RMSPE():
    modelo = cargar_modelo('modelo_73')
    X,Y=cargar_conjunto("dataset_VL_11")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scale = scaler_x.fit_transform(X)
    y_scale = scaler_y.fit_transform(Y)
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale, y_scale, train_size=0.5, test_size=0.5, random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, train_size=0.5, test_size=0.5, random_state=42)
    
    
    print("PREDICCION %")
    y_pred = modelo.predict(x_test,verbose=0)
    
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test)
    
    y_pred = np.power(10,y_pred)
    y_test = np.power(10,y_test)
    rmspe = (np.sqrt(np.mean(np.square((y_test - y_pred) / y_test)))) * 100

    print("rmspe = {}".format(rmspe))
    
generar_RMSPE()