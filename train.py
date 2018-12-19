import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Activation, Average, Dense, Flatten
from keras.layers.convolutional import AveragePooling3D, Conv3D
from keras.models import Model
from keras.optimizers import SGD
from keras.engine.input_layer import Input
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, ThresholdedReLU
from keras import regularizers
from keras.regularizers import l2

from keras import backend as K
K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')

from sklearn.metrics import classification_report, f1_score

def sequential_cnn_model(input_shape, C1, numPCAcomponents, num_classes=9, optimizer='adam'): 
    model = Sequential()
    
    model.add(Conv2D(C1, (3,3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(3*C1, (3,3), activation='relu'))
    model.add(Dropout(0.25))
    
    
    model.add(AveragePooling2D(pool_size=(1, 1), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(30*numPCAcomponents, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    return model

def train_test_model(model, X_train, y_train, X_test, y_test, target_names):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    print("Model f1_score on train data/labels:", f1_score(y_train, y_pred_train, average="weighted"), "\n")
    y_pred = model.predict(X_test)
    print("Model score on test data/labels:\n" + classification_report(y_test, y_pred, target_names=target_names))


#model = KerasClassifier(build_fn=create_model, verbose=0)
# sequential_cnn_model = sequential_cnn_model(input_shape)

# # On a 9 classes en réalité, donc une accuracy > 100/9 ~=11.11 est supérieure au hasard. 
# sequential_cnn_model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=7,
#           verbose=1)

# model = sequential_cnn_model