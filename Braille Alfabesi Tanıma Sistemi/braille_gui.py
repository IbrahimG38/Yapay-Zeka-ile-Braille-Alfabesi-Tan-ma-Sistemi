from tkinter import *
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model('models/braille_model.h5')
print("Model başarıyla yüklendi.")

# Tkinter penceresini oluştur
root = Tk()
root.title("Braille Harf Tanıma")
root.geometry("850x750")
root.configure(bg="#004d40")  # Dikkat çekici ve zarif bir mavi ton

# Başlık: Zarif ve Dikkat Çekici
title_label = Label(root, text="Braille Harf Tanıma", font=("Helvetica", 36, "bold"), fg="white", bg="#004d40")
title_label.pack(pady=30)

# Tespit Edilen Harf
detected_letter_label = Label(root, text="Tespit Edilen Harf: ", font=("Arial", 24, "bold"), bg="#004d40", fg="#ffff00")
detected_letter_label.pack(pady=10)

# Görsel Alanı
detected_image_frame = Frame(root, bg="#ffffff", highlightbackground="#ffd54f", highlightthickness=5, relief="solid")
detected_image_frame.pack(pady=20)

detected_image_label = Label(detected_image_frame, bg="#ffffff")
detected_image_label.pack(padx=15, pady=15)

# Oluşturulan Kelime Bölümü
predicted_word_label = Label(root, text="--Oluşturulan Kelime--", font=("Arial", 20, "italic"), fg="yellow",
                             bg="#004d40")
predicted_word_label.pack(pady=15)

# Görsellerin Yan Yana Sıralandığı Frame
image_frame = Frame(root, bg="#004d40")
image_frame.pack(pady=20)

# Butonlar için çerçeve
button_frame = Frame(root, bg="#004d40")
button_frame.pack(pady=20)

predicted_letters = []


# Görsel Yükleme Fonksiyonu
def upload_image():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

    if file_paths:
        for file_path in file_paths:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Görseli yenileme ve model tahmini
                img_resized = cv2.resize(img, (28, 28)).reshape(28, 28, 1) / 255.0
                prediction = model.predict(np.expand_dims(img_resized, axis=0))
                predicted_class = np.argmax(prediction, axis=1)

                # Braille harflerini tanımla
                letters = 'abcdefghijklmnopqrstuvwxyz'
                detected_letter = letters[predicted_class[0]].upper()
                predicted_letters.append(detected_letter)

                # Sonucu ekranda gösterme
                detected_letter_label.config(text=f"Tespit Edilen Harf: {detected_letter}")
                predicted_word_label.config(text="Oluşturulan Kelime: " + "".join(predicted_letters))

                # Görseli görüntüleme
                img_pil = Image.fromarray(img)
                img_resized_display = img_pil.resize((120, 120))
                img_tk_display = ImageTk.PhotoImage(img_resized_display)
                detected_image_label.config(image=img_tk_display)
                detected_image_label.image = img_tk_display

                # Görseli alt Frame'e ekle
                img_tk = ImageTk.PhotoImage(img_resized_display)
                img_label = Label(image_frame, image=img_tk, bg='#004d40')
                img_label.image = img_tk
                img_label.pack(side=LEFT, padx=10)

    else:
        messagebox.showinfo("Bilgi", "Dosya seçilmedi!")


# Resetleme Fonksiyonu
def reset():
    global predicted_letters
    predicted_letters = []
    detected_letter_label.config(text="Tespit Edilen Harf: ")
    predicted_word_label.config(text="Oluşturulan Kelime: ")
    detected_image_label.config(image="")
    for widget in image_frame.winfo_children():
        widget.destroy()


# Butonların daha estetik olması için fonksiyon
def on_hover(button, bg_color, fg_color):
    button.config(bg=bg_color, fg=fg_color)


# "Görsel Yükle" butonu
upload_button = Button(button_frame, text="Görsel Yükle", command=upload_image, font=("Arial", 14, "bold"),
                       bg='#00bcd4', fg="white", relief="flat", width=20, height=2)
upload_button.grid(row=0, column=0, padx=20)
upload_button.bind("<Enter>", lambda e: on_hover(upload_button, "#0097a7", "white"))
upload_button.bind("<Leave>", lambda e: on_hover(upload_button, "#00bcd4", "white"))

# "Resetle" butonu
reset_button = Button(button_frame, text="Resetle", command=reset, font=("Arial", 14, "bold"),
                      bg='#ff5722', fg="white", relief="flat", width=20, height=2)
reset_button.grid(row=0, column=1, padx=20)
reset_button.bind("<Enter>", lambda e: on_hover(reset_button, "#e64a19", "white"))
reset_button.bind("<Leave>", lambda e: on_hover(reset_button, "#ff5722", "white"))

# Tkinter arayüzünü başlat
root.mainloop()
































