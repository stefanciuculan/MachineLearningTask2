import os
import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import matplotlib.pyplot as plt

# Functie pentru incarcarea imaginilor dintr-un folder
def load_images_from_folder(folder, target_shape=None):
    images = []  # Lista pentru a stoca imaginile
    labels = []  # Lista pentru a stoca etichetele corespunzatoare imaginilor
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(('.jpeg', '.jpg')):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(subfolder)  # Se adauga eticheta corespunzatoare imaginii
                    else:
                        print(f"Avertisment: Incapabil să încarce {filename}")
    return images, labels

# Functie pentru extragerea caracteristicilor HOG din imagini
def extract_hog_features(images):
    features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.append(hog(img_gray, block_norm='L2-Hys'))
    return np.array(features)

# Definirea folderului cu datele de antrenare
data_folder = './train'

# Incarcarea imaginilor de antrenare si a etichetelor corespunzatoare
images, labels = load_images_from_folder(data_folder, target_shape=(200, 200))

# Extrage caracteristicile HOG din imaginile de antrenare
X_train_features = extract_hog_features(images)

# Scalarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)

# Initializarea si antrenarea clasificatorului Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train_scaled, labels)

# Definirea folderului cu datele de test
test_folder = './test_accuracy'

# Incarcarea imaginilor de test si a etichetelor corespunzatoare
test_images, test_labels = load_images_from_folder(test_folder, target_shape=(200, 200))

# Extrage caracteristicile HOG din imaginile de test
X_test_images_features = extract_hog_features(test_images)

# Scalarea datelor de test
X_test_images_scaled = scaler.transform(X_test_images_features)

# Realizeaza predictii folosind clasificatorul antrenat
y_test_images_pred = clf.predict(X_test_images_scaled)

# Evaluarea performantei pe setul de test
accuracy_test_images = accuracy_score(test_labels, y_test_images_pred)
print(f"Acuratețe pe Imaginile de Test: {accuracy_test_images * 100:.2f}%")

# Afisarea rezultatelor pentru fiecare imagine din setul de test
for i, (test_image, predicted_label) in enumerate(zip(test_images, y_test_images_pred)):
    print(f"Imagine Test {i + 1}: Predicted - {predicted_label}, Real - {test_labels[i]}")

    # Afisarea imaginii de test
    plt.figure()
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Imagine Test {i + 1}-=={predicted_label.upper()}==")
    plt.axis('off')
    plt.show()
