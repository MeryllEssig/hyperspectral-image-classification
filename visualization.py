import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import patches
import itertools
import spectral
from spectral import spy_colors

#our utils functions
import utils as u

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def reports (model, X_test,y_test, target_names):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    

    
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    
    return classification, confusion, Test_Loss, Test_accuracy


def Patch(data, height_index, width_index, PATCH_SIZE):
    #transpose_array = data.transpose((2,0,1))
    #print transpose_array.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

def createPredictedImage(X, y, model, PATCH_SIZE, height, width):
    outputs = np.zeros((height,width)) # zeroed image
    index = 0
    for i in range(0, height-PATCH_SIZE+1):
        if i % 8 == 0 or index == (height-PATCH_SIZE+1) - 1:
            u.printProgressBar(index + 1, (height-PATCH_SIZE+1) )
        index += 1
        for j in range(0, width-PATCH_SIZE+1):
            target = int(y[int(i+PATCH_SIZE/2)][int(j+PATCH_SIZE/2)])
            if target == 0 :
                continue
            else :
                image_patch=Patch(X,i,j, PATCH_SIZE)
                #print (image_patch.shape)
                X_test_image = image_patch.reshape(1, image_patch.shape[0],image_patch.shape[1], image_patch.shape[2]).astype('float32')#.reshape(1,image_patch.shape[2],image_patch.shape[0],image_patch.shape[1]).astype('float32')                                   
                prediction = (model.predict_classes(X_test_image))                         
                outputs[int(i+PATCH_SIZE/2)][int(j+PATCH_SIZE/2)] = prediction + 1
    return outputs