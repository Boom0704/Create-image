import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 데이터 경로 설정
base_dir = "D:/data/Vegetable Images"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# 데이터 전처리
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        # 이미지를 최대 20도 회전
    width_shift_range=0.2,    # 좌우로 20% 범위에서 이동
    height_shift_range=0.2,   # 상하로 20% 범위에서 이동
    shear_range=0.15,         # 뒤틀림(시어) 범위
    zoom_range=0.2,           # 확대/축소 범위
    horizontal_flip=True,     # 좌우 반전
    fill_mode='nearest'       # 빈 공간 채우기
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(128, 128), batch_size=32, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(128, 128), batch_size=32, class_mode='categorical'
)

# 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 저장 콜백
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 학습
history = model.fit(
    train_generator, validation_data=validation_generator, epochs=50, callbacks=[checkpoint, early_stopping]
)

# 테스트 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Test Loss: {test_loss:.2f}")

# 학습/검증 정확도 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 샘플 예측 시각화
sample_test_images, sample_test_labels = next(test_generator)
predictions = model.predict(sample_test_images)

for i in range(5):  # 샘플 5개 예측
    plt.imshow(sample_test_images[i])
    plt.title(f"Actual: {np.argmax(sample_test_labels[i])}, Predicted: {np.argmax(predictions[i])}")
    plt.show()
