#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:36:58 2017

@author: Andreas Georgopoulos
"""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cross_validation import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# Load Data -------------------------------------------------------------------------
train_set = pd.read_csv("train.csv").dropna().values
test_set  = pd.read_csv("test.csv").dropna().values
train_labels = train_set[:,0]
# Reshape to be [samples][pixels][width][height]
train_images = train_set[:,1:].reshape(-1,1,28,28).astype('float32')
test_set  = test_set[:,0:].reshape(-1,1,28,28).astype('float32')



# Visualise Data ----------------------------------------------------------------------
train_set_vis = pd.read_csv("train.csv")
plt.figure(figsize=(7,7))
for digit_num in range(1,70):
    plt.subplot(7,10,digit_num+1)
    grid_data = train_set_vis.iloc[digit_num,1:785].reshape(28, 28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = pyplot.get_cmap('gray'))
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig('mnist_dataset')    
plt.show() 


##########################################################################################
#################### Pre Processing: Data Augmentation & Transformation ##################
##########################################################################################

# Set Seed
seed = 7
np.random.seed(seed)
# Data Augmentation ---------------------------------------------------------------------
data_augmen = ImageDataGenerator( 
            zca_whitening=True,
		 width_shift_range=0.1,
		 height_shift_range=0.1,
		 zoom_range=0.1)
 
# Fit parameters from data
data_augmen.fit(train_images)

# Plot sample of new images
plt.figure(figsize=(7,7))
for train_image_batch, train_label_batch in data_augmen.flow(train_images, train_labels, batch_size=70):
	# create a grid of 3x3 images
    for i in range(0, 70): #(0,9)
        #pyplot.subplot(330 + 1 + i)
        plt.subplot(7,10,i+1)
        pyplot.imshow(train_image_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
        plt.xticks([])
        plt.yticks([])
	# show the plot
    pyplot.savefig('myfig')    
    pyplot.show() 
    break

# For each input image generate 1 randomly transformed image as well
train_images_new = train_images
train_labels_new = train_labels

batches = 0
for train_image_batch, train_label_batch in data_augmen.flow(train_images, train_labels, batch_size=1000):
    batches += 1
    print(batches)
    for i in range(0,len(train_image_batch)):
        train_images_new = np.concatenate((train_images_new, [train_image_batch[i]]), axis=0)
        train_labels_new = np.concatenate((train_labels_new, [train_label_batch[i]]))
    if batches >= len(train_images) / 1000:
        # break the loop because the generator loops nonstop
        break

# Save new datasets
np.save("train_images_new.npy",train_images_new)
np.save("train_labels_new.npy",train_labels_new)
#train_images_new = np.load("train_images_new.npy")
#train_labels_new = np.load("train_labels_new.npy")


# Preprocess new images and initial ones --------------------------------------------------

# Split training set into train (80%) and validation set (20%) to test in-sample-performance
training_images, validation_images, training_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.20, random_state=2)
training_images_new, validation_images_new, training_labels_new, validation_labels_new = train_test_split(train_images_new, train_labels_new, test_size=0.20, random_state=2)

# Normalize inputs from 0-255 to 0-1
training_images = training_images / 255
validation_images = validation_images / 255
test_images = test_set /255
training_images_new = training_images_new / 255
validation_images_new = validation_images_new / 255

# One hot encode outputs
training_labels = np_utils.to_categorical(training_labels)
validation_labels = np_utils.to_categorical(validation_labels)
training_labels_new = np_utils.to_categorical(training_labels_new)
validation_labels_new = np_utils.to_categorical(validation_labels_new)
num_classes = validation_labels.shape[1]




##########################################################################################
################################## Reporting - Metrics ###################################
##########################################################################################

def classifaction_report_csv(report):
	"""    
	Input: Classification report produced by sklearn
	Output: Datframe of classification metrics
	"""
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe
    

    
##########################################################################################
################################# CNN Model Evaluation ###################################
##########################################################################################


# Model 1 --------------------------------------------------------------------------------

def base_model_2(): 
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (1, 28, 28), padding='same', activation='relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_2 = base_model_2()
# Fit the model
model_2.fit(training_images, training_labels, validation_data=(validation_images, validation_labels), epochs=10, batch_size=200, verbose=1)
# Save model
model_2.save('model_2.h5')
# Load model
model_2 = load_model('model_2.h5')

# Final evaluation of the model on validation set
scores_2 = model_2.evaluate(validation_images, validation_labels, verbose=1)
print("Baseline Error: %.2f%%" % (100-scores_2[1]*100))

# Predict on test set
pred_cnn_2 = model_2.predict_classes(test_images, batch_size=64)
# Create csv file for kaggle submission
submission_cnn_keras_2 = pd.DataFrame({
    "ImageId": np.arange(1, pred_cnn_2.shape[0] + 1),
    "Label": pred_cnn_2.astype(int)
})
submission_cnn_keras_2.to_csv("submission_cnn_keras_2.csv", index=False)

# Confusion Matrix
validation_true, validation_pred_model_2 = validation_labels.argmax(1), model_2.predict_classes(validation_images, batch_size=64).astype(int)
confusion_matrix(validation_true, validation_pred_model_2)
np.savetxt("conf_matrix_model_2.csv", confusion_matrix(validation_true, validation_pred_model_2),delimiter=",")

# Classification report
print(classification_report(validation_true, validation_pred_model_2))
report_model_2 = classification_report(validation_true, validation_pred_model_2)
report_model_2_df = classifaction_report_csv(report_model_2)
report_model_2_df.to_csv('classification_report_model_2.csv', index = False)


# Model 1 with data augmentation ---------------------------------------------------------
# Model 1 --------------------------------------------------------------------------------


model_2_new = base_model_2()
# Fit the model on the new training images (after data augmentation)
model_2_new.fit(training_images_new, training_labels_new, validation_data=(validation_images_new, validation_labels_new), epochs=10, batch_size=200, verbose=1)
# Save model
model_2_new.save('model_2_new.h5')
# Load model
model_2_new = load_model('model_2_new.h5')

# Final evaluation of the model on validation set
scores_2_new = model_2_new.evaluate(validation_images_new, validation_labels_new, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores_2_new[1]*100))

# Predict on test set
pred_cnn_2_new = model_2_new.predict_classes(test_images, batch_size=64)
# Create csv file for kaggle submission
submission_cnn_keras_2_new = pd.DataFrame({
    "ImageId": np.arange(1, pred_cnn_2_new.shape[0] + 1),
    "Label": pred_cnn_2_new.astype(int)
})
submission_cnn_keras_2_new.to_csv("submission_cnn_keras_2_new.csv", index=False)

# Confusion Matrix on Validation Data
validation_true, validation_pred_model_2_new = validation_labels_new.argmax(1), model_2_new.predict_classes(validation_images_new, batch_size=64).astype(int)
confusion_matrix(validation_true, validation_pred_model_2_new)
np.savetxt("conf_matrix_model_2_new.csv", confusion_matrix(validation_true, validation_pred_model_2_new),delimiter=",")

# Classification report
print(classification_report(validation_true, validation_pred_model_2_new))
report_model_2_new_df = classifaction_report_csv(classification_report(validation_true, validation_pred_model_2_new))
report_model_2_new_df.to_csv('classification_report_model_2_new.csv', index = False)




# Model 2 --------------------------------------------------------------------------------

def base_model_5(): 
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), input_shape = (1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Generate Model
model_5 = base_model_5()
# Fit the model
model_5.fit(training_images, training_labels, validation_data=(validation_images, validation_labels), epochs=10, batch_size=200, verbose=1)
# save model
model_5.save('model_5.h5')
# Load model
model_5 = load_model('model_5.h5')

# Final evaluation of the model on validation set
scores_5 = model_5.evaluate(validation_images, validation_labels, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores_5[1]*100))

# Predict on test set
pred_cnn_5 = model_5.predict_classes(test_images, batch_size=64) 
# Create csv file for kaggle submission
submission_cnn_keras_5 = pd.DataFrame({
    "ImageId": np.arange(1, pred_cnn_5.shape[0] + 1),
    "Label": pred_cnn_5.astype(int)
})
submission_cnn_keras_5.to_csv("submission_cnn_keras_5.csv", index=False)

# Confusion Matrix
validation_true, validation_pred_model_5 = validation_labels.argmax(1), model_5.predict_classes(validation_images, batch_size=64).astype(int)
confusion_matrix(validation_true, validation_pred_model_5)
np.savetxt("conf_matrix_model_5.csv", confusion_matrix(validation_true, validation_pred_model_5),delimiter=",")

# Classification report
print(classification_report(validation_true, validation_pred_model_5))
report_model_5 = classification_report(validation_true, validation_pred_model_5)
report_model_5_df = classifaction_report_csv(report_model_5)
report_model_5_df.to_csv('classification_report_model_5.csv', index = False)

 

# Model 2 with data augmentation ---------------------------------------------------------

model_5_new = base_model_5()
# Fit the model
model_5_new.fit(training_images_new, training_labels_new, validation_data=(validation_images_new, validation_labels_new), epochs=10, batch_size=200, verbose=1)
# save model
model_5_new.save('model_5_new.h5')
# Load model
model_5_new = load_model('model_5_new.h5')

# Final evaluation of the model on validation set
scores_5_new = model_5_new.evaluate(validation_images_new, validation_labels_new, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores_5_new[1]*100))

# Predict on test set
pred_cnn_5_new = model_5_new.predict_classes(test_images, batch_size=64)
# Create csv file for kaggle submission
submission_cnn_keras_5_new = pd.DataFrame({
    "ImageId": np.arange(1, pred_cnn_5_new.shape[0] + 1),
    "Label": pred_cnn_5_new.astype(int)
})
submission_cnn_keras_5_new.to_csv("submission_cnn_keras_5_new.csv", index=False)
    
# Confusion Matrix
validation_true, validation_pred_model_5_new = validation_labels_new.argmax(1), model_5_new.predict_classes(validation_images_new, batch_size=64).astype(int)
confusion_matrix(validation_true, validation_pred_model_5_new)
np.savetxt("conf_matrix_model_5_new.csv", confusion_matrix(validation_true, validation_pred_model_5_new),delimiter=",")

# Clasification report
print(classification_report(validation_true, validation_pred_model_5_new))
report_model_5_new = classification_report(validation_true, validation_pred_model_5_new)
report_model_5_new_df = classifaction_report_csv(report_model_5_new)
report_model_5_new_df.to_csv('classification_report_model_5_new.csv', index = False)



# Model 3 --------------------------------------------------------------------------------
def base_model_very_deep(): 
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (1, 28, 28), activation='relu', padding='same'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_deep = base_model_very_deep()
# Fit the model
model_deep.fit(training_images, training_labels, validation_data=(validation_images, validation_labels), epochs=10, batch_size=128, verbose=1)
# save model
model_deep.save('model_deep.h5')  
# Load model
model_deep = load_model('model_deep.h5')

# Final evaluation of the model on validation set
scores_deep = model_deep.evaluate(validation_images, validation_labels, verbose=1)
print("Baseline Error: %.2f%%" % (100-scores_deep[1]*100))

# Predict on test set
pred_cnn_deep = model_deep.predict_classes(test_images, verbose=0)
# Create csv file for kaggle submission
submission_cnn_keras_deep = pd.DataFrame({
    "ImageId": np.arange(1, pred_cnn_deep.shape[0] + 1),
    "Label": pred_cnn_deep.astype(int)
})
submission_cnn_keras_deep.to_csv("submission_cnn_keras_deep.csv", index=False)

# Confusion Matrix
validation_true, validation_pred_model_deep = validation_labels.argmax(1), model_deep.predict_classes(validation_images, batch_size=128).astype(int)
confusion_matrix(validation_true, validation_pred_model_deep)
np.savetxt("conf_matrix_model_deep.csv", confusion_matrix(validation_true, validation_pred_model_deep),delimiter=",")

# Classification Report
print(classification_report(validation_true, validation_pred_model_deep))
report_model_very_deep = classification_report(validation_true, validation_pred_model_deep)
report_model_very_deep_df = classifaction_report_csv(report_model_very_deep)
report_model_very_deep_df.to_csv('classification_report_model_deep.csv', index = False)



# Model 3 with data augmentation ---------------------------------------------------------

model_deep_new = base_model_very_deep()
# Fit the model
model_deep_new.fit(training_images_new, training_labels_new, validation_data=(validation_images_new, validation_labels_new), epochs=8, batch_size=200, verbose=1)
# save model
model_deep_new.save('model_deep_new.h5')
# Load model
model_deep_new = load_model('model_deep_new.h5')

# Final evaluation of the model on validation set
scores_deep_new = model_deep_new.evaluate(validation_images_new, validation_labels_new, verbose=1)
print("Baseline Error: %.2f%%" % (100-scores_deep_new[1]*100))

# Predict on test set
pred_cnn_deep_new = model_deep_new.predict_classes(test_images, batch_size=64)
# Create csv file for kaggle submission
submission_cnn_keras_deep_new = pd.DataFrame({
    "ImageId": np.arange(1, pred_cnn_deep_new.shape[0] + 1),
    "Label": pred_cnn_deep_new.astype(int)
})
submission_cnn_keras_deep_new.to_csv("submission_cnn_keras_deep_new.csv", index=False)
    
# Confusion Matrix
validation_true, validation_pred_model_deep_new = validation_labels_new.argmax(1), model_deep_new.predict_classes(validation_images_new, batch_size=64).astype(int)
confusion_matrix(validation_true, validation_pred_model_deep_new)
np.savetxt("conf_matrix_model_deep_new.csv", confusion_matrix(validation_true, validation_pred_model_deep_new),delimiter=",")

# Classification Report
print(classification_report(validation_true, validation_pred_model_deep_new))
report_model_deep_new = classification_report(validation_true, validation_pred_model_deep_new)
report_model_deep_new_df = classifaction_report_csv(report_model_deep_new)
report_model_deep_new_df.to_csv('classification_report_model_deep_new.csv', index = False)





#  Ensemble Classifier of all models -----------------------------------------------------

# Extract all predictions provided by aforementioned models
predictions = [submission_cnn_keras_2, submission_cnn_keras_5_new, submission_cnn_keras_5, submission_cnn_keras_deep, submission_cnn_keras_deep_new]
ensemble_pred = submission_cnn_keras_2_new
ensemble_pred=ensemble_pred.rename(columns = {'Label':'Label_best'})
for i in predictions:
    ensemble_pred = ensemble_pred.merge(i, on='ImageId')

ensemble_pred["Label_ensemble"] = np.nan
no_col = len(ensemble_pred.columns)
for i in range(0,len(ensemble_pred)):
    c = Counter(ensemble_pred.iloc[i,1:6].values)
    # Find majority vote if not tie
    if len(Counter(ensemble_pred.iloc[i,1:6].values).most_common(1)[0]) == 2:
        ens_pred = Counter(ensemble_pred.iloc[i,1:6].values).most_common(1)[0][0].astype(int)
    else:
        ens_pred = ensemble_pred.iloc[i,1].astype(int)
    ensemble_pred.iloc[i,no_col-1] = ens_pred.astype(int)

pred_cnn_ensemble = ensemble_pred.iloc[:,no_col-1].values
# Create csv file for kaggle submission
submission_cnn_ensemble = pd.DataFrame({
    "ImageId": np.arange(1, pred_cnn_ensemble.shape[0] + 1),
    "Label": pred_cnn_ensemble.astype(int)
})
submission_cnn_ensemble.to_csv("submission_cnn_ensemble.csv", index=False) # kaggle score 0.99286


##########################################################################################
######################################### END ############################################
##########################################################################################


