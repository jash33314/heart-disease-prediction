import numpy as np
import keras
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Flatten,Input,Conv2D,MaxPooling2D
from keras.utils import to_categorical
from keras import backend as k

(x_train,y_train),(x_test,y_test)=mnist.load_data()
img_rows,img_cols=28,28
if k.image_data_format=="channels_first":
    x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape=(1,img_rows,img_cols)
else:
    x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    input_shape=(img_rows,img_cols,1)
x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


input_layer=Input(shape=input_shape)
con1=Conv2D(64,(3,3),activation="relu")(input_layer)
con2=Conv2D(32,(3,3),activation="relu")(con1)
pool=MaxPooling2D(pool_size=(3,3))(con2)
drop_out=Dropout(0.5)(pool)
flatten=Flatten()(drop_out)
dense=Dense(250,activation="sigmoid")(flatten)
output_layer=Dense(10,activation="softmax")(dense)
model=Model(inputs=[input_layer],outputs=[output_layer])

model.compile(optimizer=keras.optimizers.Adadelta(),loss=keras.losses.categorical_crossentropy,metrics=["accuracy"])
model.fit(x_train,y_train,epochs=12,batch_size=500)
score=model.evaluate(x_test,y_test,verbose=0)
print("test loss",score[0])
print("test accuracy",score[1])
