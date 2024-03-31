#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/benrajukoshy/Montagesimulation/blob/main/toolwear.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import shutil
import os

directory_to_delete = '/content/ROI_images'

# Check if the directory exists
if os.path.exists(directory_to_delete):
    # Remove the directory and its contents
    shutil.rmtree(directory_to_delete)
    print(f"Directory '{directory_to_delete}' and its contents have been deleted.")
else:
    print(f"Directory '{directory_to_delete}' does not exist.")


# In[2]:


# loading the necessary libraries
import numpy as np
import os
import random
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras.utils import to_categorical
from google.colab.patches import cv2_imshow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split

# Load and preprocess the data
import pandas as pd
import random
from urllib.request import urlopen, urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
import cv2
from sklearn.metrics import precision_recall_curve

# Import the pre-trained models
from tensorflow.keras.applications import ResNet50, MobileNetV2

# Define callbacks and training parameters
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

# Build the model
from keras.models import Sequential, Model, load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

pip install opencv-python


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


#Initalising the cropping parameters
roi_x1 = 1
roi_y1 = 430
roi_x2 = 1600
roi_y2 = 830


# In[5]:


filename = "S14.jpg"
filename = filename.split("S")[1]
filename = filename.split(".")[0]
print(filename)


# In[6]:


# Define the directory containing the images and the output directory
image_directory = "/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/Trainingsdaten"
output_directory = "/content/ROI_images"
os.makedirs(output_directory, exist_ok=True)
class_data = pd.read_csv(f'{image_directory}/input.csv')
print(class_data.head())


def create_roi(image_path, output_directory, class_names):
    # Read the image
    image = cv2.imread(image_path)

    # Crop the image to the specified ROI
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    # Create a subdirectory for the class
    class_directory = os.path.join(output_directory, class_names)
    os.makedirs(class_directory, exist_ok=True)

    # Get the filename without the path
    filename = os.path.basename(image_path)

    # Save the ROI image in the class-specific directory
    output_path = os.path.join(class_directory, filename)
    cv2.imwrite(output_path, roi)

# Process each image in the dataset
#for class_names in os.listdir(image_directory):
    #class_directory = os.path.join(image_directory, class_names)
    #print(class_directory)
if os.path.isdir(image_directory):
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            class_id = filename.split("S")[1]
            class_id = class_id.split(".")[0]
            #class_name = class_data.loc[class_data['Klasse'] == class_id].iloc[0]
            class_name = class_data.loc[class_data['Schneide']==int(class_id),'Klasse'].item()
            print(class_id,class_name)
            image_path = os.path.join(image_directory, filename)
            create_roi(image_path, output_directory, class_name)


# In[75]:


# Define the directory containing the images and the output directory
image_directory = "/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/Testdatensatz"
output_directory = "/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/DATA/Test"
os.makedirs(output_directory, exist_ok=True)
class_data = pd.read_csv("/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/Trainingsdaten/input.csv")
print(class_data.head())


def create_roi(image_path, output_directory, class_names):
    # Read the image
    image = cv2.imread(image_path)

    # Crop the image to the specified ROI
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    # Create a subdirectory for the class
    class_directory = os.path.join(output_directory, class_names)
    os.makedirs(class_directory, exist_ok=True)

    # Get the filename without the path
    filename = os.path.basename(image_path)

    # Save the ROI image in the class-specific directory
    output_path = os.path.join(class_directory, filename)
    cv2.imwrite(output_path, roi)

# Process each image in the dataset
#for class_names in os.listdir(image_directory):
    #class_directory = os.path.join(image_directory, class_names)
    #print(class_directory)
if os.path.isdir(image_directory):
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            class_id = filename.split("S")[1]
            class_id = class_id.split(".")[0]
            #class_name = class_data.loc[class_data['Klasse'] == class_id].iloc[0]
            class_name = class_data.loc[class_data['Schneide']==int(class_id),'Klasse'].item()
            print(class_id,class_name)
            image_path = os.path.join(image_directory, filename)
            create_roi(image_path, output_directory, class_name)


# In[7]:


# Define the directory containing the cropped ROI images
cropped_images_directory = "/content/ROI_images"

def display_random_image(images_directory):
    class_names = os.listdir(images_directory)
    random_class = random.choice(class_names)

    class_directory = os.path.join(images_directory, random_class)
    random_image = random.choice([f for f in os.listdir(class_directory) if f.endswith(".jpg")])
    image_path = os.path.join(class_directory, random_image)

    image = cv2.imread(image_path)

    size = os.path.getsize(image_path) / 1024  # Size in KB
    shape = image.shape

    print(f"Class Name: {random_class}")
    print(f"Image Path: {image_path}")
    print(f"Image Size: {size:.2f} KB")
    print(f"Image Shape: {shape}")

    cv2_imshow(image)

# Display information for a random cropped image
display_random_image(cropped_images_directory)


# In[8]:


import os
import random
import cv2
from google.colab.patches import cv2_imshow  # Required for displaying images in Google Colab

# Define the directory containing the cropped ROI images
cropped_images_directory = "/content/ROI_images"

def display_random_image(images_directory):
    class_names = os.listdir(images_directory)

    if not class_names:
        print("Error: No class directories found in the specified directory.")
        return

    random_class = random.choice(class_names)

    class_directory = os.path.join(images_directory, random_class)
    image_files = [f for f in os.listdir(class_directory) if f.endswith(".jpg")]

    if not image_files:
        print(f"Error: No images found in {class_directory}.")
        return

    random_image = random.choice(image_files)
    image_path = os.path.join(class_directory, random_image)

    image = cv2.imread(image_path)

    size = os.path.getsize(image_path) / 1024  # Size in KB
    shape = image.shape

    print(f"Class Name: {random_class}")
    print(f"Image Path: {image_path}")
    print(f"Image Size: {size:.2f} KB")
    print(f"Image Shape: {shape}")

    cv2_imshow(image)

# Display information for a random cropped image
display_random_image(cropped_images_directory)


# In[10]:


# Define the desired image shape
IMAGE_SHAPE = (224, 224)

# Set the batch size for data processing during training
BATCH_SIZE = 32


# In[11]:


# Create an ImageDataGenerator for data augmentation

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,)


# Define the path to your training data directory
training_data = "/content/ROI_images"

# Create separate data generators for training and validation
train_data_generator = image_generator.flow_from_directory(
    os.path.join(training_data),
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)


# In[12]:


total_images = train_data_generator.samples
print(f"Total number of images: {total_images}")


# In[13]:


# Get a batch of data from the generator
image_batch, label_batch = next(train_data_generator)

# image_batch: A batch of images
# label_batch: A batch of labels

# to can print the shapes of the batches to verify
print("Image batch shape:", image_batch.shape)
print("Label batch shape:", label_batch.shape)


# In[14]:


# Select a random index within the batch
index_to_display = random.randint(0, len(train_data_generator) - 1)

# Retrieve the selected image batch and label batch from the generator
selected_image_batch, selected_label_batch = train_data_generator[index_to_display]

# Extract a single image from the batch (assuming batch size is 1)
selected_image = selected_image_batch[0]

# Extract the corresponding label index (assuming batch size is 1)
selected_label_index = np.argmax(selected_label_batch[0])  # Convert one-hot to index

# Map the label index to the class name
selected_label = selected_label_index

# Display the image and label
plt.imshow(selected_image)
plt.title(f"Label: {selected_label}")
plt.axis('off')
plt.show()


# In[18]:


# Define the class names (categories) associated with your data
class_names = ['Mittel', 'Defekt', 'Neuwertig']
# Define the number of examples to display per class
examples_per_class = 32  # You can adjust this number as needed

# Create a dictionary to store images for each class
class_images = {class_name: [] for class_name in class_names}

# Iterate through the data generator to collect images for each class
for i in range(len(train_data_generator)):
    image_batch, label_batch = train_data_generator[i]
    for j in range(len(label_batch)):
        label_index = np.argmax(label_batch[j])
        class_name = class_names[label_index]
        if len(class_images[class_name]) < examples_per_class:
            class_images[class_name].append(image_batch[j])

# Display examples from each class
for class_name, images in class_images.items():
    print(f"Class: {class_name}")
    for i, image in enumerate(images):
        plt.subplot(len(class_names), examples_per_class, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


# In[19]:


# Create a dictionary to store the counts for each class
class_counts = {class_name: 0 for class_name in class_names}

# Iterate through the data generator to count the images per class
for i in range(len(train_data_generator)):
    _, label_batch = train_data_generator[i]
    for j in range(len(label_batch)):
        label_index = np.argmax(label_batch[j])
        class_name = class_names[label_index]
        class_counts[class_name] += 1

# Print the total number of images in each class
for class_name, count in class_counts.items():
    print(f"Class: {class_name}, Total Images: {count}")


# In[20]:


# Define the number of batches to consider for splitting
num_batches = len(train_data_generator)

# Initialize empty lists to store images and labels
images = []
labels = []

# Load all batches from the generator into lists
for _ in range(num_batches):
    image_batch, label_batch = next(train_data_generator)
    images.append(image_batch)
    labels.append(label_batch)

# Convert the lists to NumPy arrays
X = np.vstack(images)
y = np.vstack(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the training and test sets
print(f"Training set size: {len(X_train)} images")
print(f"Test set size: {len(X_test)} images")


# In[22]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
model = tf.keras.Sequential([feature_extractor])
model.summary()


# In[24]:


MobileNetV2=tf.keras.applications.mobilenet_v2.MobileNetV2
model_arch=MobileNetV2()
model_arch.summary()


# In[25]:


model_1 = tf.keras.Sequential([feature_extractor ])
model_1.add(tf.keras.layers.Dense(3, activation='softmax'))
model_1.summary()


# In[26]:


model_1.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'] )


# In[27]:


# Define the number of training epochs
epochs = 25

# Train the model using training data and validation data
history_1 = model_1.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))


# In[28]:


print(history_1.history.keys())
# summarize history for accuracy
plt.plot(history_1.history['accuracy'])  # Change 'acc' to 'accuracy'
plt.plot(history_1.history['val_accuracy'])  # Change 'val_acc' to 'val_accuracy'
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[29]:


# summarize history for loss
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[30]:


# Make predictions using the model
predicted_batch1 = model_1.predict(X_train)

# Find the predicted class indices
predicted_id1 = np.argmax(predicted_batch1, axis=-1)

# Map predicted class indices to class labels
predicted_label_batch1 = [class_names[i] for i in predicted_id1]

# Find the true class indices (ground truth)
true_id1 = np.argmax(y_train, axis=-1)

# Map true class indices to true class labels
true_label_batch1 = [class_names[i] for i in true_id1]

# Now you can print the predicted and true labels for the batch
print("Predicted Labels:", predicted_label_batch1)
print("True Labels:", true_label_batch1)


# In[31]:


#Function to plot an image
def plotImages(images1):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images1, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[32]:


img_counter=0
prob_counter=0

plt.figure(figsize=(24,16))
plt.subplots_adjust(hspace=1)

# Loop to plot images and probability distributions for predictions
for n in range(32):
  plt.subplot(4,8,n+1)
  if(n%2==0):
      plt.imshow(X_train[img_counter])
      color = "green" if predicted_id1[img_counter] == true_id1[img_counter] else "red"
      #plt.title(predicted_label_batch[img_counter].title(), color=color)
      plt.title("Pred: "+predicted_label_batch1[img_counter].title()+"\nTrue: "+true_label_batch1[img_counter], color=color)
      plt.axis('off')
      _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
      img_counter=img_counter+1

  if(n%2==1):

      plt.title("Dense Layer Output (Logits)", color="Black")

      data=[predicted_batch1[prob_counter][0],predicted_batch1[prob_counter][1],predicted_batch1[prob_counter][2]]
      classes=['Mittel','Neuwertig', 'Defect']
      plt.xticks(rotation='vertical',)
      plt.tick_params(axis='both', which='major', labelsize=12)

      plt.bar(class_names,data)
      prob_counter=prob_counter+1


# In[48]:


# Define the path to your training data directory
testing_data = "/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/Testdatensatz/"

parent_dir = "/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/DATA"
os.makedirs(parent_dir, exist_ok=True)

#shutil.copytree("/content/ROI_images","/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/DATA/Train")
#shutil.copytree("/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/Testdatensatz","/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/DATA/Test")

# Create separate data generators for training and validation
test_data_generatormob = image_generator.flow_from_directory(parent_dir, target_size=IMAGE_SHAPE)


# In[49]:


for test_image_batchmob, test_label_batchmob in test_data_generatormob :
  print("Image batch shape: ", test_image_batchmob.shape)
  print("Label batch shape: ", test_label_batchmob.shape)
  break


# In[50]:


predicted_batchtest1 = model_1.predict(test_image_batchmob)
predicted_idtest1 = np.argmax(predicted_batchtest1, axis=-1)
label_idtest1 = np.argmax( test_label_batchmob, axis=-1)


# In[51]:


im = 5  # Define the index of the image to analyze

# Make predictions using the model
predictions = model.predict(test_image_batchmob[[im]])  # Get predictions for the selected image

print("Predicted Class Probabilities:", predictions)  # Print the predicted class probabilities
print("Class Names:", class_names)  # Print the names of the classes

# Print the predicted class index for the selected image
print("Predicted Class Index:",predicted_idtest1[im])

# Check if the predicted class is different from the true class
if predicted_idtest1[im] != label_idtest1[im]:
    print("\033[31mModel Prediction Class:", predicted_idtest1[im], "\033[0m")
    # If different, print the predicted class label in red
else:
    print("Model Prediction Class:", predicted_idtest1[im])
    # If the same, print the predicted class label

# Print the true class label for the selected image
print("True Class Label:", label_idtest1[im])

# Display the selected image
plt.imshow(image_batch[im])


# In[52]:


img_counter = 0  # Initialize a counter for iterating through images
prob_counter = 0  # Initialize a counter for iterating through probabilities

# Create a figure for plotting
plt.figure(figsize=(24, 16))
plt.subplots_adjust(hspace=1)  # Adjust spacing between subplots

# Loop to plot images and probability distributions for predictions
for n in range(15):
    plt.subplot(4, 8, n + 1)  # Create subplots in a 4x8 grid

    if n % 2 == 0:
        # Plot the test image
        plt.imshow(test_image_batchmob[img_counter])

        # Determine the color of the title based on prediction correctness
        color = "green" if predicted_idtest1[img_counter] == label_idtest1[img_counter] else "red"

        # Set the title with predicted and true labels
        # Map class indices to class names
        # Inside your loop, you can display class names using class_indices
        plt.title("Pred: " + class_names[predicted_idtest1[img_counter]].title() + "\nTrue: " + class_names[label_idtest1[img_counter]].title(), color=color)
        plt.axis('off')  # Turn off axis for the image
        _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")  # Super title for the entire plot

        img_counter += 1  # Increment the image counter

    if n % 2 == 1:
        # Title for the Dense Layer Output (Logits) plot
        plt.title("Dense Layer Output (Logits)", color="Black")

        # Extract the predicted probabilities for each class
        data = [predicted_batchtest1[prob_counter][0], predicted_batchtest1[prob_counter][1], predicted_batchtest1[prob_counter][2]]

        # Rotate x-axis labels vertically and adjust tick parameters
        plt.xticks(rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Create a bar plot to display the predicted probabilities
        plt.bar(class_names, data)

        prob_counter += 1  # Increment the probability counter

# The code above creates a visualization with images on the left and bar plots of predicted probabilities on the right.
# The titles of the images indicate whether the prediction is correct (green) or incorrect (red).
# The bar plots show the predicted probabilities for each class.


# In[53]:


import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()

con_mat_df=tf.math.confusion_matrix(labels =label_idtest1 , predictions = predicted_idtest1)
figure = plt.figure(figsize=(5, 5))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[54]:


from sklearn.metrics import classification_report
print(classification_report(label_idtest1, predicted_idtest1, zero_division=0))


# In[55]:


#Define Number of Classes
NUM_CLASSES = 3


# In[56]:


#Loading Pretrained ResNet50 Model
from keras import applications
base_model =applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (224,224,3))


# In[57]:


# Define Top Layers for Custom Classification
x = base_model.output
# Global Average Pooling Layer
x = GlobalAveragePooling2D()(x)
# Dense Layer for Custom Classification
x = Dropout(0.7)(x)
# Final Output Layer
predictions = Dense(NUM_CLASSES, activation= 'softmax')(x)
# Create the Custom Model
model_2 = Model(inputs = base_model.input, outputs = predictions)


# In[58]:


#Configures the model for training by specifying the optimization algorithm ('adam'), the loss function ('categorical_crossentropy'), and the evaluation metric ('accuracy').
from keras.optimizers import SGD, Adam
adam = Adam(lr=0.0001)
model_2.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history_2 = model_2.fit(X_train, y_train, epochs=10, batch_size = 32, validation_data=(X_test, y_test))


# In[60]:


model_2.summary()


# In[64]:


# Print the keys available in the history_2 object
print(history_2.history.keys())
plt.plot(history_2.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[65]:


# summarize history for loss
# Print the keys available in the history_2 object
print(history_2.history.keys())

plt.plot(history_2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

# Add a legend to the plot to differentiate between 'train' and 'validation'
# Specify the position of the legend in the 'upper left' corner

plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[66]:


# Make predictions using the model
predicted_batch2 = model_2.predict(X_train)
predicted_id2 = np.argmax(predicted_batch2, axis=-1)
predicted_label_batch2 = [class_names[i] for i in predicted_id2]
true_id2 = np.argmax(y_train, axis=-1)
true_label_batch2 = [class_names[i] for i in true_id2]
print("Predicted Labels:", predicted_label_batch2)
print("True Labels:", true_label_batch2)


# In[67]:


img_counter = 0
prob_counter = 0

plt.figure(figsize=(24, 16))
plt.subplots_adjust(hspace=1)
num_images = min(32, len(X_train))

for n in range(num_images):

    plt.subplot(4, 8, n + 1)
    if n % 2 == 0:
        plt.imshow(X_train[img_counter])
        color = "green" if predicted_id2[img_counter] == true_id2[img_counter] else "red"
        plt.title("Pred: " + predicted_label_batch2[img_counter].title() + "\nTrue: " + true_label_batch2[img_counter], color=color)
        plt.axis('off')
        _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
        img_counter += 1

    if n % 2 == 1:
        plt.title("Dense Layer Output (Logits)", color="Black")
        data = [predicted_batch2[prob_counter][0], predicted_batch2[prob_counter][1], predicted_batch2[prob_counter][2]]
        plt.xticks(rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.bar(class_names, data)
        prob_counter += 1


# In[82]:



test_data_generatorRES = image_generator.flow_from_directory("/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/Test", target_size=IMAGE_SHAPE)
test_data_generatorRES.class_indices


# In[84]:


for test_image_batchRES, test_label_batchRES in test_data_generatorRES :
  print("Image batch shape: ", test_image_batchRES.shape)
  print("Label batch shape: ", test_label_batchRES.shape)
  break


# In[85]:


predicted_batchtest2 = model_2.predict(test_image_batchRES)
predicted_idtest2 = np.argmax(predicted_batchtest2, axis=-1)
label_idtest2 = np.argmax( test_label_batchRES, axis=-1)


# In[86]:


im = 3  # Define the index of the image to analyze

# Make predictions using the model
predictions = model_2.predict(test_image_batchRES[[im]])
print("Predicted Class Probabilities:", predictions)
print("Class Names:", class_names)

print("Predicted Class Index:", predicted_idtest2[im])
if predicted_idtest2[im] != label_idtest2[im]:
    print("\033[31mModel Prediction Class:", predicted_idtest2[im], "\033[0m")
else:
    print("Model Prediction Class:", predicted_idtest2[im])
print("True Class Label:",label_idtest2[im])
plt.imshow(test_image_batchRES[im])


# In[87]:


class_names = ['Mittel', 'Defect', 'Neuwertig']


# In[88]:


img_counter = 0
prob_counter = 0
plt.figure(figsize=(24, 16))
plt.subplots_adjust(hspace=1)

# Loop to plot images and probability distributions for predictions
for n in range(14):
    plt.subplot(4, 8, n + 1)
    if n % 2 == 0:
        plt.imshow(test_image_batchRES[img_counter])

        color = "green" if predicted_idtest2[img_counter] == label_idtest2[img_counter] else "red"
        plt.title("Pred: " + class_names[predicted_idtest2[img_counter]].title() + "\nTrue: " + class_names[label_idtest2[img_counter]].title(), color=color)
        plt.axis('off')
        _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
        img_counter += 1

    if n % 2 == 1:
        plt.title("Dense Layer Output (Logits)", color="Black")
        data = [predicted_batchtest2[prob_counter][0], predicted_batchtest2[prob_counter][1], predicted_batchtest2[prob_counter][2]]

        plt.xticks(rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.bar(class_names, data)
        prob_counter += 1


# In[89]:


import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()

con_mat_df=tf.math.confusion_matrix(predictions = predicted_idtest2, labels = label_idtest2)
figure = plt.figure(figsize=(5, 5))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[90]:


from sklearn.metrics import classification_report
print(classification_report(predicted_idtest2, label_idtest2,  zero_division=0))


# In[91]:


# Define the ImageDataGenerator with augmentation parameters
Augmentationgenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5,         # Rotate the image by up to 10 degrees
    width_shift_range=0.05,     # Shift the width by up to 10% of the image's width
    height_shift_range=0.05,    # Shift the height by up to 10% of the image's height
    shear_range=0.05,          # Shear the image by up to 15 degrees
    zoom_range=0.05,            # Zoom in or out by up to 10%
    channel_shift_range=50.,  # Adjust color channels by up to 50
    horizontal_flip=True       # Flip the image horizontally
)


# In[95]:


# Define the path to the directory containing the images for a specific class
chosen_image = random.choice(os.listdir('/content/ROI_images/Defekt'))
image_path = '/content/ROI_images/Defekt/' + chosen_image
image = np.expand_dims(plt.imread(image_path),0)
# Define the path to the directory where augmented images should be saved
aug_iterater = Augmentationgenerator.flow(image, save_to_dir='/content/ROI_images/Defekt', save_prefix='aug-image-', save_format='jpg')
# Now you can run the augmented image generation code
aug_images = [next(aug_iterater)[0].astype(np.uint8) for i in range(250)]


# In[98]:


plotImages(aug_images)


# In[97]:


chosen_image = random.choice(os.listdir('/content/ROI_images/Neuwertig'))
image_path = '/content/ROI_images/Neuwertig/' + chosen_image
image = np.expand_dims(plt.imread(image_path),0)
aug_iterater = Augmentationgenerator.flow(image, save_to_dir='/content/ROI_images/Neuwertig', save_prefix='aug-image-', save_format='jpg')
aug_images = [next(aug_iterater)[0].astype(np.uint8) for i in range(250)]


# In[99]:


plotImages(aug_images)


# In[100]:


# Define the desired image shape
IMAGE_SHAPE = (224, 224)
# Set the batch size for data processing during training
BATCH_SIZE = 32


# In[101]:


# Create an ImageDataGenerator for data augmentation
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,)
# Define the path to your training data directory
training_data_withaug = "/content/ROI_images"
# Create separate data generators for training and validation
train_dataaug_generator = image_generator.flow_from_directory(
    os.path.join(training_data_withaug),
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)


# In[102]:


total_images = train_dataaug_generator.samples
print(f"Total number of images: {total_images}")


# In[103]:


# Get a batch of data from the generator
image_batch, label_batch = next(train_dataaug_generator)
print("Image batch shape:", image_batch.shape)
print("Label batch shape:", label_batch.shape)


# In[104]:


# Select a random index within the batch
index_to_display = random.randint(0, len(train_dataaug_generator) - 1)
selected_image_batch, selected_label_batch = train_dataaug_generator[index_to_display]
selected_image = selected_image_batch[0]
selected_label_index = np.argmax(selected_label_batch[0])
selected_label = selected_label_index

# Display the image and label
plt.imshow(selected_image)
plt.title(f"Label: {selected_label}")
plt.axis('off')
plt.show()


# In[105]:


class_names = ['Mittel', 'Defect', 'Neuwertig']
examples_per_class = 32
class_images = {class_name: [] for class_name in class_names}

for i in range(len(train_dataaug_generator)):
    image_batch, label_batch = train_dataaug_generator[i]
    for j in range(len(label_batch)):
        label_index = np.argmax(label_batch[j])
        class_name = class_names[label_index]
        if len(class_images[class_name]) < examples_per_class:
            class_images[class_name].append(image_batch[j])


for class_name, images in class_images.items():
    print(f"Class: {class_name}")
    for i, image in enumerate(images):
        plt.subplot(len(class_names), examples_per_class, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


# In[108]:


# Create a dictionary to store the counts for each class
class_counts = {class_name: 0 for class_name in class_names}

# Iterate through the data generator to count the images per class
for i in range(len(train_dataaug_generator)):
    _, label_batch = train_dataaug_generator[i]
    for j in range(len(label_batch)):
        label_index = np.argmax(label_batch[j])
        class_name = class_names[label_index]
        class_counts[class_name] += 1

# Print the total number of images in each class
for class_name, count in class_counts.items():
    print(f"Class: {class_name}, Total Images: {count}")


# In[109]:


# Define the number of batches to consider for splitting
num_batches = len(train_dataaug_generator)

# Initialize empty lists to store images and labels
images = []
labels = []

# Load all batches from the generator into lists
for _ in range(num_batches):
    image_batch, label_batch = next(train_dataaug_generator)
    images.append(image_batch)
    labels.append(label_batch)

# Convert the lists to NumPy arrays
X = np.vstack(images)
y = np.vstack(labels)

# Split the data into training and test sets
X_trainaug, X_testaug, y_trainaug, y_testaug = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the training and test sets
print(f"Training set size: {len(X_train)} images")
print(f"Test set size: {len(X_test)} images")


# In[110]:


print(X_trainaug.shape)
print(X_testaug.shape)
print(y_trainaug.shape)
print(y_testaug.shape)


# In[111]:


model_3 = tf.keras.Sequential([feature_extractor ])
model_3.add(tf.keras.layers.Dense(3, activation='softmax'))
model_3.summary()


# In[112]:


model_3.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'] )


# In[113]:


# Train the model using training data and validation data
history_1withaug = model_3.fit(X_trainaug, y_trainaug, epochs=10 , validation_data=(X_testaug, y_testaug))


# In[114]:


print(history_1withaug.history.keys())
# summarize history for accuracy
plt.plot(history_1withaug.history['accuracy'])
plt.plot(history_1withaug.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[115]:


# summarize history for loss
plt.plot(history_1withaug.history['loss'])
plt.plot(history_1withaug.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[116]:


# Make predictions using the model
predicted_batch3 = model_3.predict(X_trainaug)
predicted_id3 = np.argmax(predicted_batch3, axis=-1)
predicted_label_batch3 = [class_names[i] for i in predicted_id3]
true_id3 = np.argmax(y_trainaug, axis=-1)
true_label_batch3 = [class_names[i] for i in true_id3]

# Now you can print the predicted and true labels for the batch
print("Predicted Labels:", predicted_label_batch3)
print("True Labels:", true_label_batch3)


# In[117]:


img_counter=0
prob_counter=0

plt.figure(figsize=(24,16))
plt.subplots_adjust(hspace=1)
# Loop to plot images and probability distributions for predictions
for n in range(32):
  plt.subplot(4,8,n+1)
  if(n%2==0):
      plt.imshow(X_trainaug[img_counter])
      color = "green" if predicted_id3[img_counter] == true_id3[img_counter] else "red"
      plt.title("Pred: "+predicted_label_batch3[img_counter].title()+"\nTrue: "+true_label_batch3[img_counter], color=color)
      plt.axis('off')
      _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
      img_counter=img_counter+1

  if(n%2==1):
      plt.title("Dense Layer Output (Logits)", color="Black")

      data=[predicted_batch3[prob_counter][0],predicted_batch3[prob_counter][1],predicted_batch3[prob_counter][2]]
      classes=['Container_Groß','Container_Klein', 'Lader']
      plt.xticks(rotation='vertical',)
      plt.tick_params(axis='both', which='major', labelsize=12)

      plt.bar(class_names,data)
      prob_counter=prob_counter+1


# In[119]:


# Create an ImageDataGenerator for data augmentation
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,)
# Define the path to your training data directory
testing_data_MobAug = "/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/Test"
# Create separate data generators for training and validation
test_data_generator_MobAug = image_generator.flow_from_directory(testing_data_MobAug, target_size=IMAGE_SHAPE)


# In[120]:


for test_image_batch_MobAug, test_label_batch_MobAug in test_data_generator_MobAug :
  print("Image batch shape: ", test_image_batch_MobAug.shape)
  print("Label batch shape: ", test_label_batch_MobAug.shape)
  break


# In[121]:


predicted_batch_MobAug = model_3.predict(test_image_batch_MobAug)
predicted_id_MobAug = np.argmax(predicted_batch_MobAug, axis=-1)
label_id_MobAug = np.argmax( test_label_batch_MobAug, axis=-1)


# In[122]:


im = 5  # Define the index of the image to analyze

predictions_MobAug = model.predict(test_image_batch_MobAug[[im]])
print("Predicted Class Probabilities:", predictions_MobAug)
print("Class Names:", class_names)
print("Predicted Class Index:", predicted_id_MobAug[im])


if predicted_id_MobAug[im] != label_id_MobAug[im]:
    print("\033[31mModel Prediction Class:", predicted_id_MobAug[im], "\033[0m")
else:
    print("Model Prediction Class:",predicted_id_MobAug[im])

print("True Class Label:",label_id_MobAug[im])
# Display the selected image
plt.imshow((test_image_batch_MobAug[im]))


# In[123]:


img_counter = 0
prob_counter = 0

# Create a figure for plotting
plt.figure(figsize=(24, 16))
plt.subplots_adjust(hspace=1)

# Loop to plot images and probability distributions for predictions
for n in range(15):
    plt.subplot(4, 8, n + 1)

    if n % 2 == 0:
        plt.imshow(test_image_batch_MobAug[img_counter])

        color = "green" if predicted_id_MobAug[img_counter] == label_id_MobAug[img_counter] else "red"
        plt.title("Pred: " + class_names[predicted_id_MobAug[img_counter]].title() + "\nTrue: " + class_names[label_id_MobAug[img_counter]].title(), color=color)
        plt.axis('off')
        _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")

        img_counter += 1

    if n % 2 == 1:
        plt.title("Dense Layer Output (Logits)", color="Black")
        data = [predicted_batch_MobAug[prob_counter][0], predicted_batch_MobAug[prob_counter][1], predicted_batch_MobAug[prob_counter][2]]


        plt.xticks(rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.bar(class_names, data)
        prob_counter += 1



# In[125]:


import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()

con_mat_df=tf.math.confusion_matrix(labels = label_id_MobAug , predictions = predicted_id_MobAug)
figure = plt.figure(figsize=(5, 5))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[126]:


from sklearn.metrics import classification_report
print(classification_report(label_id_MobAug, predicted_id_MobAug, zero_division=0))


# In[127]:


#Define Number of Classes
NUM_CLASSES = 3


# In[128]:


# Define Top Layers for Custom Classification
x = base_model.output
# Global Average Pooling Layer
x = GlobalAveragePooling2D()(x)
# Dense Layer for Custom Classification
x = Dropout(0.7)(x)
# Final Output Layer
predictions = Dense(NUM_CLASSES, activation= 'softmax')(x)
# Create the Custom Model
model_4 = Model(inputs = base_model.input, outputs = predictions)


# In[129]:


#Configures the model for training by specifying the optimization algorithm ('adam'), the loss function ('categorical_crossentropy'), and the evaluation metric ('accuracy').
from keras.optimizers import SGD, Adam
adam = Adam(lr=0.0001)
model_4.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[130]:


history_4 = model_4.fit(X_trainaug, y_trainaug, epochs=10, validation_data=(X_testaug, y_testaug))


# In[131]:


# Print the keys available in the history_2 object

print(history_4.history.keys())
plt.plot(history_4.history['accuracy'])
plt.plot(history_4.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# Display the accuracy plot on the screen
plt.show()


# In[132]:


# summarize history for loss
# Print the keys available in the history_2 object
print(history_4.history.keys())

plt.plot(history_4.history['loss'])
plt.plot(history_4.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

# Add a legend to the plot to differentiate between 'train' and 'validation'
# Specify the position of the legend in the 'upper left' corner

plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[133]:


# Make predictions using the model
predicted_batch_ResAug = model_4.predict(X_trainaug)
predicted_id_ResAug = np.argmax(predicted_batch_ResAug, axis=-1)
predicted_label_batch_ResAug = [class_names[i] for i in predicted_id_ResAug]
true_id_ResAug = np.argmax(y_trainaug, axis=-1)
true_label_batch_ResAug = [class_names[i] for i in true_id_ResAug]

# Now you can print the predicted and true labels for the batch
print("Predicted Labels:", predicted_label_batch_ResAug)
print("True Labels:", true_label_batch_ResAug)


# In[134]:


img_counter = 0
prob_counter = 0

plt.figure(figsize=(24, 16))
plt.subplots_adjust(hspace=1)
num_images = min(32, len((X_trainaug)))  # Ensure you don't exceed the available images

for n in range(num_images):

    plt.subplot(4, 8, n + 1)
    if n % 2 == 0:
        plt.imshow(X_trainaug[img_counter])
        color = "green" if predicted_id_ResAug[img_counter] == true_id_ResAug[img_counter] else "red"
        plt.title("Pred: " + predicted_label_batch_ResAug[img_counter].title() + "\nTrue: " + true_label_batch_ResAug[img_counter], color=color)
        plt.axis('off')
        _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
        img_counter += 1

    if n % 2 == 1:
        plt.title("Dense Layer Output (Logits)", color="Black")
        data = [predicted_batch_ResAug[prob_counter][0], predicted_batch_ResAug[prob_counter][1], predicted_batch_ResAug[prob_counter][2]]
        plt.xticks(rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.bar(class_names, data)
        prob_counter += 1


# In[136]:


# Define the path to your training data directory
testing_data_ResAug = "/content/drive/MyDrive/Datensatz Seminararbeit WS23-24/Test"

# Create separate data generators for training and validation
test_data_generator_ResAug = image_generator.flow_from_directory(testing_data_ResAug, target_size=IMAGE_SHAPE)


# In[137]:


for test_image_batch_ResAug, test_label_batch_ResAug in test_data_generator_ResAug :
  print("Image batch shape: ", test_image_batch_ResAug.shape)
  print("Label batch shape: ", test_label_batch_ResAug.shape)
  break


# In[138]:


predicted_batch_testResAug = model_4.predict(test_image_batch_ResAug)
predicted_id_testResAug = np.argmax(predicted_batch_testResAug, axis=-1)
label_id_testResAug = np.argmax( test_label_batch_ResAug, axis=-1)


# In[139]:


im = 1  # Define the index of the image to analyze

# Make predictions using the model
predictions_ResAug = model.predict(test_image_batch_ResAug [[im]])  # Get predictions for the selected image

print("Predicted Class Probabilities:", predictions_ResAug )  # Print the predicted class probabilities
print("Class Names:", class_names)  # Print the names of the classes

# Print the predicted class index for the selected image
print("Predicted Class Index:", predicted_id_ResAug [im])

# Check if the predicted class is different from the true class
if predicted_id_testResAug[im] !=label_id_testResAug[im]:
    print("\033[31mModel Prediction Class:", predicted_id_testResAug[im], "\033[0m")
    # If different, print the predicted class label in red
else:
    print("Model Prediction Class:", predicted_id_testResAug[im])
    # If the same, print the predicted class label

# Print the true class label for the selected image
print("True Class Label:", label_id_testResAug[im])

# Display the selected image
plt.imshow(image_batch[im])


# In[140]:


img_counter = 0
prob_counter = 0
plt.figure(figsize=(24, 16))
plt.subplots_adjust(hspace=1)


for n in range(15):
    plt.subplot(4, 8, n + 1)

    if n % 2 == 0:
        plt.imshow(test_image_batch_ResAug[img_counter])
        color = "green" if predicted_id_testResAug[img_counter] == label_id_testResAug[img_counter] else "red"
        plt.title("Pred: " + class_names[predicted_id_testResAug[img_counter]].title() + "\nTrue: " + class_names[label_id_testResAug[img_counter]].title(), color=color)
        plt.axis('off')
        _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")  # Super title for the entire plot
        img_counter += 1

    if n % 2 == 1:
        plt.title("Dense Layer Output (Logits)", color="Black")

        data = [predicted_batch_testResAug[prob_counter][0],predicted_batch_testResAug[prob_counter][1],predicted_batch_testResAug[prob_counter][2]]

        plt.xticks(rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.bar(class_names, data)
        prob_counter += 1


# In[141]:


import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()

con_mat_df=tf.math.confusion_matrix(labels = label_id_testResAug , predictions = predicted_id_testResAug )
figure = plt.figure(figsize=(5, 5))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[142]:


from sklearn.metrics import classification_report
print(classification_report(label_id_testResAug, predicted_id_testResAug, zero_division=0))


# In[ ]:




