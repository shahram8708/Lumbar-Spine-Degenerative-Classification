# %% [code]
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-06-06T12:03:28.806571Z","iopub.execute_input":"2024-06-06T12:03:28.807048Z","iopub.status.idle":"2024-06-06T12:03:41.711245Z","shell.execute_reply.started":"2024-06-06T12:03:28.807006Z","shell.execute_reply":"2024-06-06T12:03:41.709170Z"}}
import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

train_df['image'] = train_df['image_path'].apply(preprocess_image)
X = np.array(train_df['image'].tolist())
y = train_df[['normal_mild', 'moderate', 'severe']].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 224, 224, 3)
X_val = X_val.reshape(-1, 224, 224, 3)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_acc}")

test_df['image'] = test_df['image_path'].apply(preprocess_image)
X_test = np.array(test_df['image'].tolist())
X_test = X_test.reshape(-1, 224, 224, 3)

predictions = model.predict(X_test)

submission_df = pd.DataFrame(predictions, columns=['normal_mild', 'moderate', 'severe'])
submission_df['row_id'] = test_df['row_id']
submission_df = submission_df[['row_id', 'normal_mild', 'moderate', 'severe']]

submission_df.to_csv('submission.csv', index=False)