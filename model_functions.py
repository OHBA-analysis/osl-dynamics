import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import ops

import importlib

import taser_functions
import misc

taser_functions = importlib.reload(taser_functions)
misc = importlib.reload(misc)

log_print = misc.log_print


def create_model(mini_batch_length, nchans, npriors, SL_tmp_cov_mat):
    # log_print(f"LOG: scope in function is {ops.get_default_graph()._distribution_strategy_stack}", "red")

    if True:
        bi_di_inf = 1
        bi_di_model = 0
        nunits = 50
        # Build the network!
        inputs_layer = layers.Input(shape=(mini_batch_length, nchans), name='MEG_input')
        dropout_layer = layers.Dropout(0.5)(inputs_layer)

        if bi_di_inf == 1:
            print("using bi-directional RNNs")
            inf_output_fw = tf.keras.layers.GRU(int(nunits / 1),  # number of units
                                                return_sequences=True,
                                                name='uni_INF_GRU_FW')
            inf_output_bw = tf.keras.layers.GRU(int(nunits / 1),  # number of units
                                                return_sequences=True,
                                                name='uni_INF_GRU_BW',
                                                go_backwards=True)

            output = tf.keras.layers.Bidirectional(inf_output_fw,
                                                   backward_layer=inf_output_bw,
                                                   merge_mode='concat')(dropout_layer)
        else:

            output = tf.keras.layers.GRU(int(nunits / 1),  # number of units
                                         return_sequences=True,
                                         name='uni_INF_GRU')(dropout_layer)

        # Affine transformations from output of RNN to mu and sigma for each alpha:
        dense_layer_mu = tf.keras.layers.Dense(npriors, activation='linear')(output)
        dense_layer_sigma = tf.keras.layers.Dense(npriors, activation='linear')(output)

        # Generate a sample via the RT.
        # alpha_ast = layers.Lambda(taser_functions.sampling,
        #                           name='alpha_ast')(z_mean=dense_layer_mu, z_log_var=dense_layer_sigma)
        alpha_ast = taser_functions.SamplingLayer()([dense_layer_mu, dense_layer_sigma])

        if bi_di_model == 1:
            model_output_fw = tf.keras.layers.GRU(int(nunits / 1),  # number of units
                                                  return_sequences=True,
                                                  name='uni_model_mu_GRU_FW')
            model_output_bw = tf.keras.layers.GRU(int(nunits / 1),  # number of units
                                                  return_sequences=True,
                                                  name='uni_model_mu_GRU_BW',
                                                  go_backwards=True)

            model_output = tf.keras.layers.Bidirectional(model_output_fw,
                                                         backward_layer=model_output_bw,
                                                         merge_mode='concat')(alpha_ast)
        else:
            model_output = tf.keras.layers.GRU(int(nunits / 1),  # number of units
                                               return_sequences=True,
                                               name='uni_model_mu_GRU')(alpha_ast)

        model_dense_layer_mu = tf.keras.layers.Dense(npriors, activation='linear', name='model_mu')(model_output)
        model_dense_layer_sigma = tf.keras.layers.Dense(npriors, activation='linear', name='model_sigma')(model_output)

        # Instantiate the model:
        model = tf.keras.Model(inputs=[inputs_layer],
                               outputs=[dense_layer_mu, dense_layer_sigma, alpha_ast, model_dense_layer_mu,
                                        model_dense_layer_sigma])

        # Save the model architecture before doing anything else.
        model_as_json = model.to_json()
        with open('model.json', "w") as json_file:
            json_file.write(model_as_json)

        weight = 1.0

        # Construct your custom loss by feeding in the tensors which we made in the model.
        loss = taser_functions.my_beautiful_custom_loss(alpha_ast,
                                                        inputs_layer,
                                                        tf.constant(npriors),
                                                        tf.constant(nchans),
                                                        SL_tmp_cov_mat,
                                                        dense_layer_mu,
                                                        dense_layer_sigma,
                                                        model_dense_layer_mu,
                                                        model_dense_layer_sigma,
                                                        tf.constant(weight), tf.constant(mini_batch_length))

        # Add loss to model
        model.add_loss(loss)

        # Now compile the model with an optimizer
        opt = tf.keras.optimizers.Adam(lr=0.0008, clipnorm=0.1)
        model.compile(optimizer=opt)

    return model
