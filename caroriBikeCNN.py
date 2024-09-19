import numpy as np
import os
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Görüntülerin yüklendiği klasör yolları
folder_paths = [
    {"path": "/content/drive/MyDrive/archive/Car-Bike-Dataset/Bike", "arr": []},
    {"path": "/content/drive/MyDrive/archive/Car-Bike-Dataset/Car", "arr": []}
]

# Görüntüleri yükleme süreci
selected_images = []  # Seçilen resimlerin saklanacağı liste
labels = []

for index, folder in enumerate(folder_paths):
    print(folder["path"])
    files = os.listdir(folder["path"])
    random.shuffle(files)  # Dosyaları karıştır
    count = 0  # Seçilen resim sayısını tutmak için sayaç
    for filename in files:
        if filename.endswith(('.jpg', '.png', '.jpeg')) and count < 200:
            with Image.open(os.path.join(folder["path"], filename)) as image_to_display:
                img = ImageOps.grayscale(image_to_display).resize((64, 64))
                selected_images.append(np.array(img))
                labels.append(index)  # 0 for Bike, 1 for Car
                count += 1


# Seçilen resimlerin gösterimi
num_samples = len(selected_images)  # Seçilen resim sayısı
rows = int(np.ceil(num_samples / 10))  # Satır sayısını hesapla
fig, axes = plt.subplots(rows, 10, figsize=(20, 2 * rows))  # Daha büyük bir genişlik ve azaltılmış yükseklik ayarla
axes = axes.flatten()  # Eksenleri tek boyuta indirge
for i, ax in enumerate(axes):
    if i < num_samples:
        ax.imshow(selected_images[i].reshape(64, 64))
        ax.axis('off')
    else:
        ax.axis('off')  # Kullanılmayan subplot'ları gizle

plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Subplotlar arası boşlukları minimize et
plt.show()

# Resimlerin numpy dizisine dönüştürülmesi ve normalizasyon
X = np.array(selected_images).reshape(-1, 64, 64, 1) / 255.0
y = np.array(labels)

# Verileri eğitim ve test seti olarak ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model tanımı ve eğitimi
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64, 1)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

def evaluate_and_visualize_model(model, X, y, num_samples):
    predictions = model.predict(X)
    predicted_labels = (predictions > 0.5).astype(int)

    num_columns = 10  # Satır başına 10 resim
    rows = int(np.ceil(num_samples / num_columns))
    fig, axes = plt.subplots(rows, num_columns, figsize=(20, 2 * rows))  # Genişlik artırıldı, yükseklik azaltıldı
    axes = axes.flatten()  # Eksenleri tek boyuta indirge

    for i, ax in enumerate(axes):
        if i < len(X):
            ax.imshow(X[i].reshape(64, 64), interpolation='nearest')
            predicted_label = 'Car' if predicted_labels[i] ==1 else 'Bike'
            actual_label = 'Car' if y[i] ==1 else 'Bike'
            # Başlık yerine etiketleri kullanarak, resmin üzerine yazmayacak şekilde ayarla
            ax.text(1, 1, f'Prediction: {predicted_label}\nTrue: {actual_label}', color='black', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            ax.axis('off')
        else:
            ax.axis('off')  # Kullanılmayan subplot'ları gizle

    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Subplotlar arası boşlukları azalt
    plt.show()

# Seçilen tüm resimlerin tahminlerini görselleştir
evaluate_and_visualize_model(model, X, y, len(X))

history = model.fit(X_train, Y_train, epochs=15, batch_size=64, validation_data=(X_test, Y_test))

# Eğitim ve doğrulama kaybını ve başarı puanını grafik üzerinde göster
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Test seti üzerinde modelin nihai başarı puanını ve kaybını yazdır
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

def plot_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

# Modelin tahminlerini al ve ikili eşik uygula
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

# Confusion matrix ve classification report'u görselleştir ve yazdır
plot_confusion_matrix(Y_test, predicted_labels)
print_classification_report(Y_test, predicted_labels)
