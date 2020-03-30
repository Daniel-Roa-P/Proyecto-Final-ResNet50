# Daniel Alejandro Roa Palacios - 20171020077

import numpy as np
import os
import time
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve  

# Cargar el data de imagenes
PATH = os.getcwd()
# definir la direccion del las imagenes 
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

lista_imagenes=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)

	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
        
		lista_imagenes.append(x)

data_imagenes = np.array(lista_imagenes)

print (data_imagenes.shape)
data_imagenes=np.rollaxis(data_imagenes,1,0)
print (data_imagenes.shape)
data_imagenes=data_imagenes[0]
print (data_imagenes.shape)


# Definir el numero de clases
numero_clases = 3
numero_muestras = data_imagenes.shape[0]
etiquetas = np.ones((numero_muestras,),dtype='int64')

etiquetas[0:210]=0
etiquetas[210:420]=1
etiquetas[420:]=2

names = ['Aviones','Barcos','Aviones']

# convertir las clases en categorias

Y = np_utils.to_categorical(etiquetas, numero_clases)

# Barajar el dataset

x,y = shuffle(data_imagenes,Y, random_state=2)

# entrenar y testear el dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2)

###########################################################################################################################

# modelo 1
# Entrenar el clasificador

image_input = Input(shape=(224, 224, 3))

model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(numero_clases, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Tiempode entrenamiento %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

###########################################################################################################################

# Afinar el ResNet50

model = ResNet50(weights='imagenet',include_top=False)
model.summary()
last_layer = model.output

x = GlobalAveragePooling2D()(last_layer)
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)

out = Dense(numero_clases, activation='softmax',name='output_layer')(x)

custom_resnet_model2 = Model(inputs=model.input, outputs=out)

custom_resnet_model2.summary()

for layer in custom_resnet_model2.layers[:-6]:
	layer.trainable = False

custom_resnet_model2.layers[-1].trainable

custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = custom_resnet_model2.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

############################################################################################
import matplotlib.pyplot as plt

# Visualizacion de perdida y presicion

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('numero de epocas')
plt.ylabel('Perdidas')
plt.title('perdidas de entrenamiento vs valor de perdida')
plt.grid(True)
plt.legend(['train','val'])


plt.figure(4,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('numero de epocas')
plt.ylabel('precision')
plt.title('precision de entrenamiento vs valor de precison')
plt.grid(True)
plt.legend(['train','val'],loc=4)