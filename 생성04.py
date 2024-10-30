import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, Layer, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# 데이터 경로 설정
image_dir = "D:/data/Vegetable Images/test/Carrot"
print("데이터 경로가 설정되었습니다.")

# 데이터 증강 및 전처리
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.1)
])

def load_and_preprocess_image(path, img_size=(128, 128)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0  # [0, 1] 범위로 정규화
    return data_augmentation(image)

# 이미지 파일 경로 리스트 생성
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
print(f"{len(image_paths)}개의 이미지 파일이 발견되었습니다.")

# 데이터셋 생성
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
print("데이터셋이 생성되고 전처리가 완료되었습니다.")

# VAE 모델 설정
latent_dim = 128
input_shape = (128, 128, 3)

# 인코더 정의
class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(128, 3, activation='relu', strides=2, padding='same')
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(256, 3, activation='relu', strides=2, padding='same')
        self.bn3 = BatchNormalization()
        self.conv4 = Conv2D(512, 3, activation='relu', strides=2, padding='same')
        self.flatten = Flatten()
        self.dense = Dense(512, activation='relu')
        self.dropout = Dropout(0.3)
        self.z_mean = Dense(latent_dim)
        self.z_log_var = Dense(latent_dim)

    def call(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dropout(self.dense(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

# 샘플링 레이어 정의
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# 디코더 정의
class Decoder(Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dense = Dense(16 * 16 * 512, activation='relu')
        self.reshape = Reshape((16, 16, 512))
        self.convT1 = Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.convT2 = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')
        self.bn2 = BatchNormalization()
        self.convT3 = Conv2DTranspose(64, 3, activation='relu', strides=1, padding='same')
        self.bn3 = BatchNormalization()
        self.convT4 = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')
        self.convT5 = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')

    def call(self, x):
        x = self.dense(x)
        x = self.reshape(x)
        x = self.bn1(self.convT1(x))
        x = self.bn2(self.convT2(x))
        x = self.bn3(self.convT3(x))
        x = self.convT4(x)  
        return self.convT5(x)

# VAE 모델 정의
class VAE(Model):
    def __init__(self, encoder, decoder, kl_weight=0.005, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.sampling = Sampling()
        self.decoder = decoder
        self.kl_weight = kl_weight

    def train_step(self, x):
        if isinstance(x, tuple):
            x = x[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = self.sampling((z_mean, z_log_var))
            x_recon = self.decoder(z)

            # 손실 계산 (MSE + BCE 사용)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_recon))
            reconstruction_loss *= input_shape[0] * input_shape[1]
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': total_loss}

    def call(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling((z_mean, z_log_var))
        return self.decoder(z)

# 인코더와 디코더 인스턴스 생성
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)

# VAE 모델 인스턴스 생성
vae = VAE(encoder, decoder, kl_weight=0.005)

# 학습률 스케줄러 정의
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=10000,
    decay_rate=0.9
)

# 모델 컴파일
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

# 모델 학습
history = vae.fit(dataset, epochs=50)

# 학습 손실 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 잠재 공간에서 이미지 생성 및 시각화 함수
def plot_latent_images(decoder, latent_dim, input_shape, n=5, scale=3.0, figsize=20):
    figure = np.zeros((input_shape[0] * n, input_shape[1] * n, input_shape[2]))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim))
            # 당근의 길쭉한 형태를 반영하기 위해 첫 두 차원에 강한 특징 부여
            z_sample[0, :2] = [xi * 1.5, yi * 0.5]
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0]
            figure[i * input_shape[0]: (i + 1) * input_shape[0],
                   j * input_shape[1]: (j + 1) * input_shape[1]] = digit

    # 시각화
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure)
    plt.axis('off')
    plt.show()

# 이미지 생성 및 시각화
plot_latent_images(decoder, latent_dim, input_shape)
