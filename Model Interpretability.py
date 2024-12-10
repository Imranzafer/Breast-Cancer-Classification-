import os
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

# Step 1: Download and extract the dataset from Kaggle
!kaggle datasets download -d nazmulhassan/breast-histopathology-images

# Unzip the dataset
import zipfile

with zipfile.ZipFile('breast-histopathology-images.zip', 'r') as zip_ref:
    zip_ref.extractall('breast_histopathology')

# Step 2: Prepare the dataset for training
# Define paths for the dataset
base_dir = 'breast_histopathology'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Step 3: Build the model using Transfer Learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=test_generator,
    validation_steps=len(test_generator),
    epochs=10
)

# Save the model
model.save('breast_cancer_model.h5')

# Step 4: SHAP Interpretability
explainer = shap.DeepExplainer(model, train_generator[0][0])
shap_values = explainer.shap_values(test_generator[0][0])

# Plot SHAP values for the first test image
shap.image_plot(shap_values, test_generator[0][0])

# Step 5: Grad-CAM for Visual Interpretability
def grad_cam(input_model, img_array, layer_name):
    preds = input_model.predict(img_array)
    class_idx = np.argmax(preds[0])

    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(input_model.output[:, class_idx], layer_output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([input_model.input], [pooled_grads, layer_output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_array])

    for i in range(pooled_grads_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# Load a test image from the test set
img_path = 'path_to_single_test_image.jpg'  # Change to a valid test image path
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img_array = np.expand_dims(img, axis=0)
img_array /= 255.0

# Apply Grad-CAM
heatmap = grad_cam(model, img_array, 'block5_conv3')

# Display the heatmap
plt.matshow(heatmap)
plt.show()

# Superimpose the heatmap on the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
cv2.imshow('Grad-CAM', superimposed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
