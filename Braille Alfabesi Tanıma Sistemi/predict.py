import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model('models/braille_model.h5')  # Modelin yolu
print("Model başarıyla yüklendi.")


# Yeni bir görsel üzerinde tahmin yapma fonksiyonu
def predict_braille_image(model, image_path):
    # Görseli okuma (gri tonlamada)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Görseli 28x28 boyutuna getir
    img = cv2.resize(img, (28, 28))

    # Görseli normalleştir (0-1 arası)
    img = img.flatten() / 255.0

    # Görseli uygun boyuta getir (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    # Model ile tahmin yap
    prediction = model.predict(img)

    # Tahmin edilen sınıfı al
    predicted_class = np.argmax(prediction)

    # Sonucu Braille harfi olarak döndür
    return chr(predicted_class + ord('a'))


# Test etmek için bir resim yolu verin
image_path = 'images/a_test_image.png'  # Buraya test etmek istediğiniz görselin yolunu yazın
predicted_letter = predict_braille_image(model, image_path)
print(f"Tahmin edilen harf: {predicted_letter}")
