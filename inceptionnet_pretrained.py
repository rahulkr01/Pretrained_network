from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import data_prepr
from keras import metrics
from keras.optimizers import Adam
import numpy as np

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 3 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator( data_utils.get_batch_data(), epochs=2,steps_per_epoch=2000) 


# let's visualize layer names and layer indices

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

model.save('pretrained_inceptionnetV3.h5')  

test_batch_size=64
total_test_batch= 500
avg=0

predicted_y=np.array([])
actual_y=np.array([])

for i in range(total_test_batch):
	evalX,evalY=data_utils.get_eval_data()
	loss,acc = model.evaluate(evalX,evalY, batch_size=64)
	y1=model.predict(evalX,batch_size=64)
	y1=np.argmax(y1,axis=1)
	y2=np.argmax(evalY,axis=1)
	predicted_y=np.hstack((predicted_y,y1))
	actual_y=np.hstack((actual_y,y2))

	print("loss: ",loss)
	print("acc: ",acc)
	avg+=acc
print("avg: ", avg/total_test_batch)
arr=np.vstack((actual_y,predicted_y)).T
np.savetxt('prettrained_predicted_labels.txt',arr)
