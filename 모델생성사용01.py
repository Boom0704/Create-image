import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 커스텀 클래스 정의
@tf.keras.saving.register_keras_serializable()
class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', strides=2, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.z_mean = tf.keras.layers.Dense(latent_dim)
        self.z_log_var = tf.keras.layers.Dense(latent_dim)

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

@tf.keras.saving.register_keras_serializable()
class Decoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dense = tf.keras.layers.Dense(16 * 16 * 512, activation='relu')
        self.reshape = tf.keras.layers.Reshape((16, 16, 512))
        self.convT1 = tf.keras.layers.Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same')
        self.convT2 = tf.keras.layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')
        self.convT3 = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')
        self.convT4 = tf.keras.layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')

    def call(self, x):
        x = self.dense(x)
        x = self.reshape(x)
        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)
        return x

# 모델 경로 설정
encoder_path = "encoder_model.keras"
decoder_path = "decoder_model.keras"

# 저장된 모델 불러오기
encoder = tf.keras.models.load_model(encoder_path, compile=False)
decoder = tf.keras.models.load_model(decoder_path, compile=False)

# 잠재 공간에서 이미지 생성 및 시각화 함수
def plot_generated_images(decoder, n=10, latent_dim=128, figsize=20, scale=5.0):
    input_shape = (128, 128, 3)  # 이미지 크기 설정
    figure = np.zeros((input_shape[0] * n, input_shape[1] * n, 3))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.random.normal(0, 1, (1, latent_dim))  # 잠재 공간에서 무작위 샘플링
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0]
            figure[i * input_shape[0]: (i + 1) * input_shape[0],
                   j * input_shape[1]: (j + 1) * input_shape[1]] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure)
    plt.axis('off')
    plt.show()

# 이미지 생성 및 시각화
plot_generated_images(decoder, n=5, latent_dim=128, figsize=20)
