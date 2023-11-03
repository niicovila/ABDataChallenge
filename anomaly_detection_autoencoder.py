import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K

from ReadData import read_data
# Assuming your data is loaded as 'data' with shape (x, 10, 5000)
data = read_data("C:\Users\nicol\Desktop\AbDataChallenge\abdataset1_barcelona_temp_seq.csv")
# Define LSTM-VAE architecture
latent_dim = 2  # Number of latent dimensions for VAE
input_shape = (10, 5000)  # Shape of each sequence

# Encoder
inputs = Input(shape=input_shape)
encoded = LSTM(64)(inputs)  # LSTM layer acts as the encoder
z_mean = Dense(latent_dim)(encoded)
z_log_var = Dense(latent_dim)(encoded)

# Reparameterization trick for VAE
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoded = LSTM(64, return_sequences=True)(z)  # LSTM layer acts as the decoder
decoded_output = Dense(5000)(decoded)

# VAE model
vae = Model(inputs, decoded_output)

# Loss function for VAE
def vae_loss(inputs, decoded_output):
    x = K.flatten(inputs)
    decoded_output = K.flatten(decoded_output)
    xent_loss = mean_squared_error(x, decoded_output)
    kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)

# Reshape the data for model input
data_reshaped = data.reshape(-1, 10, 5000)

# Train the VAE model
vae.fit(data_reshaped, data_reshaped, epochs=50, batch_size=32)

# Encode the input data to obtain latent representations
encoder = Model(inputs, z_mean)
latent_representation = encoder.predict(data_reshaped)

# Calculate anomaly scores for each sequence
reconstructed_data = vae.predict(data_reshaped)
mse = np.mean(np.square(data_reshaped - reconstructed_data), axis=(1, 2))  # Mean squared error as anomaly score
