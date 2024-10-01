from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
import numpy as np
import os

model_path = '/Real-life_Deception_Detection_2016/TrainedModel.h5'

base_model = load_model(model_path)

truth_directory = '/home/cortezm/Desktop/Project IA/Real-life_Deception_Detection_2016/Clips/TruthfulFR'
lie_directory = '/home/cortezm/Desktop/Project IA/Real-life_Deception_Detection_2016/Clips/DeceptiveFR'

def load_and_preprocess_images(directory, label):
    images = []
    labels = []

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(folder_path, filename)
                    img = load_img(img_path, target_size=(299, 299))  # InceptionV3 requiere imágenes de 299x299
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)

                    images.append(img_array)
                    labels.append(label)

    return np.array(images), np.array(labels)

truth_images, truth_labels = load_and_preprocess_images(truth_directory, 1)

lie_images, lie_labels = load_and_preprocess_images(lie_directory, 0)

X = np.concatenate([truth_images, lie_images])
y = np.concatenate([truth_labels, lie_labels])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_intermediate = base_model.predict(X_train)
X_test_intermediate = base_model.predict(X_test)

X_train_intermediate = np.reshape(X_train_intermediate, (X_train_intermediate.shape[0], 1, X_train_intermediate.shape[1]))
X_test_intermediate = np.reshape(X_test_intermediate, (X_test_intermediate.shape[0], 1, X_test_intermediate.shape[1]))

print("Forma de X_train_intermediate:", X_train_intermediate.shape)
print("Forma de X_test_intermediate:", X_test_intermediate.shape)

# Create LSTM
model_lstm = Sequential()

model_lstm.add(LSTM(64, input_shape=(X_train_intermediate.shape[1], X_train_intermediate.shape[2])))
model_lstm.add(Dense(1, activation='sigmoid'))

# Compile and train LSTM
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train_intermediate, y_train, epochs=10, batch_size=32, validation_split=0.2)

model_lstm.save('/Real-life_Deception_Detection_2016/trained_lstm_model.h5')

accuracy_lstm = model_lstm.evaluate(X_test_intermediate, y_test)[1]
print(f'Accuracy on test set using LSTM: {accuracy_lstm * 100:.2f}%')
