import cv2
import os
import numpy as np


# Veri setinden resimleri ve etiketlerini yükleme fonksiyonu
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resmi 28x28 boyutuna getir
            img = cv2.resize(img, (28, 28))
            img = img.flatten() / 255.0  # Normalizasyon (0-1 arası)
            images.append(img)
            labels.append(label)
    return images, labels


# Veriyi yükleyip işlemeye yarayan ana fonksiyon
def load_braille_dataset(dataset_dir):
    images = []
    labels = []

    # Her harf için veri yükleyin
    for label, letter in enumerate('abcdefghijklmnopqrstuvwxyz'):
        folder = os.path.join(dataset_dir, letter)
        imgs, lbls = load_images_from_folder(folder, label)
        images.extend(imgs)
        labels.extend(lbls)

    # Verileri numpy array olarak döndür
    X = np.array(images)
    y = np.array(labels)

    # Veriyi 28x28x1 boyutlarına getirelim (CNN için uygun format)
    X = X.reshape(X.shape[0], 28, 28, 1)

    return X, y
