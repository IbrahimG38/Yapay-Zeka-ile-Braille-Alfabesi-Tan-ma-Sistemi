import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import load_braille_dataset
from tensorflow.keras.models import load_model

"""
# Modeli yükle
model = load_model('models/braille_model.h5')
print("Model başarıyla yüklendi.")
"""



# Veri setini yükle
dataset_dir = 'data/braille_dataset'
X, y = load_braille_dataset(dataset_dir)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN modelini oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # 26 harf
])

# Modeli derleme
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Test setinde doğruluk
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test doğruluğu: {test_accuracy * 100:.2f}%")

# Modeli kaydedelim
model.save('models/braille_model.h5')
print("Model başarıyla kaydedildi.")



