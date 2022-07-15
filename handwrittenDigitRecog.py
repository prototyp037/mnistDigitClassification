from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tensorflow as tf

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()



print(x_train.shape)


print(x_train[0])
print(y_test[0])

#preprocess da data

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


imagePreprocessor=ImageDataGenerator(rescale=1.0/255,zoom_range=0.2,width_shift_range=0.1,height_shift_range=0.1)

training_iterator=imagePreprocessor.flow(x_train,y_train,batch_size=8)
validation_iterator=imagePreprocessor.flow(x_test,y_test,batch_size=8)



#build da model

model=keras.Sequential()

convol2DstrideValue=1

model.add(keras.Input(shape=(28,28,1)))
model.add(keras.layers.Conv2D(8,3,strides=convol2DstrideValue,padding='valid',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(keras.layers.Conv2D(8,3,strides=convol2DstrideValue,padding='valid',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

epochStuffValue=4 #either between 4 and 8 is good

model.fit(training_iterator,
steps_per_epoch=len(x_train)/8,
epochs=epochStuffValue,
validation_data=validation_iterator,
validation_steps=len(x_test)/8)

model.save("mnistModel")
model=model.load("mnistModel")

def predictImage(imagePath):
    image=cv2.imread(imagePath)
    image=tf.image.resize(image,[32,32],antialias=True)
    model.predict(image)


predictImage('C:/Users/037co/Downloads/3image')





