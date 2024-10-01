from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os


base_directory = '/Real-life_Deception_Detection_2016/Clips'

truth_directory = os.path.join(base_directory, 'TruthfulFR')
lie_directory = os.path.join(base_directory, 'DeceptiveFR')

def load_and_preprocess_images(directory, label):
    images = []
    labels = []

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(folder_path, filename)
                    img = load_img(img_path, target_size=(299, 299))  # InceptionV3 requires 299x299
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)

                    images.append(img_array)
                    labels.append(label)

    return np.array(images), np.array(labels)

# Thrutful
truth_images, truth_labels = load_and_preprocess_images(truth_directory, 1)

# Lie
lie_images, lie_labels = load_and_preprocess_images(lie_directory, 0)

# Split
X = np.concatenate([truth_images, lie_images])
y = np.concatenate([truth_labels, lie_labels])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # 1 output (thruth or lie)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

model.save('/Real-life_Deception_Detection_2016/TrainedModel.h5')

# Testing
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Accuracy on test set: {accuracy * 100:.2f}%')
