from keras import layers as L
import tensorflow as tf



def autoencoder_Titan():
    """
    General info

    Input   :
    Output  :
    """

    #setup the inputs and embed them.

    light_input = L.Input((light_len,), dtype='float', name='Light_Input')
    heavy_input = L.Input((heavy_len,), dtype='float', name='Heavy_Input')

    light_embed = L.Embedding(input_dim=input_dims, output_dim=10, name='Light_Embed')(light_input)
    heavy_embed = L.Embedding(input_dim=input_dims, output_dim=10, name='Heavy_Embed')(heavy_input)

