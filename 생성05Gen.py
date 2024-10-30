import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Conv2D, Conv2DTranspose, Flatten, Reshape,
                                     LeakyReLU, BatchNormalization, Input, Dropout)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

# -------------------------------
# 1. Data Preprocessing
# -------------------------------

# 데이터 경로 설정
image_dir = "D:/data/Vegetable Images/train/Carrot"
model_save_path = "gan_generator_model_512.h5"
print("데이터 경로가 설정되었습니다.")

# 이미지 크기 설정
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3

# 데이터 전처리 함수
def load_and_preprocess_image(path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=IMG_CHANNELS)
    image = tf.image.resize(image, img_size)
    image = (image - 127.5) / 127.5  # [-1, 1] 범위로 정규화
    return image

# 이미지 파일 경로 리스트 생성
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"{len(image_paths)}개의 이미지 파일이 발견되었습니다.")

# 데이터셋 생성
BATCH_SIZE = 16  # 메모리 사용을 고려하여 배치 사이즈 조정
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
print("데이터셋이 생성되고 전처리가 완료되었습니다.")

# -------------------------------
# 2. 모델 정의
# -------------------------------

# 공통 초기화기
initializer = RandomNormal(mean=0.0, stddev=0.02)

# 생성자 모델 정의
def build_generator(latent_dim):
    model = Sequential(name="Generator")
    model.add(Input(shape=(latent_dim,)))

    # 첫 번째 Dense layer
    model.add(Dense(8 * 8 * 512, use_bias=False, kernel_initializer=initializer))
    model.add(Reshape((8, 8, 512)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # 16x16
    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # 32x32
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # 64x64
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # 128x128
    model.add(Conv2DTranspose(32, kernel_size=5, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # 256x256
    model.add(Conv2DTranspose(16, kernel_size=5, strides=2, padding='same', use_bias=False, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # 512x512
    model.add(Conv2DTranspose(IMG_CHANNELS, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh', kernel_initializer=initializer))

    return model

# 판별자 모델 정의
def build_discriminator(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    model = Sequential(name="Discriminator")
    model.add(Input(shape=input_shape))

    # 512x512
    model.add(Conv2D(16, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # 256x256
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # 128x128
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # 64x64
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # 32x32
    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # 16x16
    model.add(Conv2D(512, kernel_size=5, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

# -------------------------------
# 3. 하이퍼파라미터 설정
# -------------------------------

latent_dim = 100
epochs = 100
sample_interval = 20  # 이미지 샘플링 주기
save_interval = 50    # 모델 저장 주기

# -------------------------------
# 4. 모델 생성 및 옵티마이저 설정
# -------------------------------

generator = build_generator(latent_dim)
discriminator = build_discriminator()

# 옵티마이저 설정
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

# 손실 함수
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# -------------------------------
# 5. 학습 루프
# -------------------------------

@tf.function
def train_step(real_images):
    # 랜덤 노이즈 생성
    noise = tf.random.normal([BATCH_SIZE, latent_dim])

    # 레이블 생성
    real_labels = tf.ones((BATCH_SIZE, 1))
    fake_labels = tf.zeros((BATCH_SIZE, 1))

    # 생성자 학습
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 생성자에 의해 생성된 이미지
        generated_images = generator(noise, training=True)

        # 판별자에 의해 예측된 값
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # 손실 계산
        gen_loss = cross_entropy(real_labels, fake_output)
        real_loss = cross_entropy(real_labels, real_output)
        fake_loss = cross_entropy(fake_labels, fake_output)
        disc_loss = real_loss + fake_loss

    # 기울기 계산
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 옵티마이저 적용
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss, gen_loss

# -------------------------------
# 6. 이미지 생성 함수
# -------------------------------

def sample_images(epoch, n=4):
    noise = np.random.normal(0, 1, (n * n, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # [0, 1] 범위로 변환

    fig, axs = plt.subplots(n, n, figsize=(n*3, n*3))
    count = 0
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis("off")
            count += 1
    plt.tight_layout()
    plt.show()

# -------------------------------
# 7. 모델 저장 함수
# -------------------------------

def save_model(epoch):
    generator.save(f"gan_generator_model_512_epoch_{epoch}.h5")
    discriminator.save(f"gan_discriminator_model_512_epoch_{epoch}.h5")
    print(f"모델이 epoch {epoch}에 저장되었습니다.")

# -------------------------------
# 8. 학습 시작
# -------------------------------

for epoch in range(1, epochs + 1):
    for real_images in dataset:
        disc_loss, gen_loss = train_step(real_images)

    # 주기적으로 손실 출력 및 이미지 샘플링
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch} [D loss: {disc_loss.numpy():.4f}] [G loss: {gen_loss.numpy():.4f}]")
        sample_images(epoch)

    # 주기적으로 모델 저장
    if epoch % save_interval == 0:
        save_model(epoch)

    # 진행 상황 출력
    if epoch % 1 == 0:
        print(f"Epoch {epoch} completed.")

# -------------------------------
# 9. 저장된 모델을 이용한 이미지 생성
# -------------------------------

def generate_images_from_saved_model(model_path, n=4):
    model = load_model(model_path)
    noise = np.random.normal(0, 1, (n * n, latent_dim))
    generated_images = model.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # [0, 1] 범위로 변환

    fig, axs = plt.subplots(n, n, figsize=(n*3, n*3))
    count = 0
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis("off")
            count += 1
    plt.tight_layout()
    plt.show()

# 사용 예시:
# generate_images_from_saved_model("gan_generator_model_512_epoch_5000.h5")
