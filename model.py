import os

import tensorflow as tf

from neurons.LIFmodel.LIFneuron import LIF
from neurons.SleepyLeakLIF.StdLeakLIF import StdLeakLIF, StdLeakyIntegrator, IntegratorCell

from auxiliary import SpikeCountMetric, SpikeRegularization

# tf.random.set_seed(52)


def create_model(neuron_type, input_shape, hidden_units, num_hidden_layers, output_shape, dropout, fixed=False,
                 recurrence=False, leaky_output=False, weight_regu=0.):

    inputs = tf.keras.layers.Input(shape=input_shape)

    hidden_layers = {}
    dropout_layers = {}
    state_v_dict = {}
    for i in range(num_hidden_layers):
        if i == 0:
            dropout_layers[i] = tf.keras.layers.Dropout(dropout)(inputs)
            n_in = inputs.shape[-1]
        else:
            dropout_layers[i] = tf.keras.layers.Dropout(dropout)(hidden_layers[i - 1])
            n_in = hidden_units

        if neuron_type == 'LIF':
            spiking_neuron = LIF(n_in=n_in, n_rec=hidden_units, tau=5, thr=1, dt=1, fixed=fixed, recurrence=recurrence)
        elif neuron_type == 'Std':
            spiking_neuron = StdLeakLIF(n_in=n_in, n_rec=hidden_units, tau=100., thr=1, dt=1, recurrence=recurrence,
                                        w_regu=weight_regu)
        else:
            raise Exception("The neuron model is unknown or ill-defined.")
        hidden_layers[i], state_v_dict[i] = tf.keras.layers.RNN(spiking_neuron, return_sequences=True,
                                                                name=f'HL{i}')(dropout_layers[i])

    dropout_last = tf.keras.layers.Dropout(dropout)(hidden_layers[num_hidden_layers - 1])

    if not leaky_output:
        output = tf.keras.layers.RNN(IntegratorCell(n_in=hidden_units, n_rec=output_shape, fixed=fixed,
                                                    w_regu=weight_regu),
                                     return_sequences=False,
                                     name='output')(dropout_last)
    else:
        output = tf.keras.layers.RNN(StdLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, dt=1.0, tau=10.,
                                                        w_regu=weight_regu),
                                     return_sequences=False,
                                     name='output')(dropout_last)

    model = tf.keras.models.Model(inputs=inputs, outputs=[output, *hidden_layers.values(), *state_v_dict.values()])

    return model


def compile_model(model, lr, eagerly, batch_size, spike_regu_factor):
    metrics = dict()
    metrics['output'] = "sparse_categorical_accuracy"
    for l in model.layers:
        if l.name[0:2] == 'HL':
            metrics[l.name] = SpikeCountMetric(batch_size=batch_size)

    losses = {'output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}
    for l in model.layers:
        if l.name[0:2] == 'HL':
            losses[l.name] = SpikeRegularization(spike_regu_factor)

    model.compile(
            loss=losses,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=metrics,
            run_eagerly=eagerly,
        )
    
    return model


def save_model(model, saving_path):
    model.save(saving_path)
    model.save(os.path.join(saving_path, "allinone.h5"))
