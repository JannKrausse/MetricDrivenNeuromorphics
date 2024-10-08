import tensorflow as tf
import numpy as np


@tf.function
def spike_share(y, spikes):
    # y not used
    size = tf.size(spikes)
    return tf.reduce_sum(spikes) / tf.cast(size, tf.float32)


class SpikeCountMetric:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    @tf.function
    def __call__(self, y, spikes):
        # y not used
        return tf.reduce_sum(spikes) / tf.cast(self.batch_size, tf.float32)


class ModelEnergy:
    def __init__(self, device):
        self.device = device
        self.get_energy_values(device)

    def get_energy_values(self, device):
        if device == 'loihi':
            self.energy_per_synops = (23.6 + 3.5) * 1e-12
            self.energy_per_neuron = 81e-12
        elif device == 'spinnaker':
            self.energy_per_synops = 13.3e-9
            self.energy_per_neuron = 26e-9
        elif device == 'spinnaker2':
            self.energy_per_synops = 450 * 1e-12
            self.energy_per_neuron = 2.19e-9
        else:
            raise Exception(f"Device {self.device} is unknown.")

    @staticmethod
    def calculate_operations(model, data, nonzero_weight_ratios):
        """
        TODO:
        1) consider pruned layer
        2) consider event-triggered updating
        3) current implementation assumes equal connectivity between all neurons
        """
        output = model(data)

        batch_size, _, in_channels = data.shape

        spikes_per_neuron_dict = dict()
        weights_per_neuron_dict = dict()
        neurons_dict = dict()

        for i, l in enumerate(model.layers):
            if l.name[0:5] == 'input':
                neurons_dict['in'] = in_channels

                spikes_per_neuron_dict['in'] = np.sum(data != 0) / (batch_size * in_channels)
            elif l.name[0:2] == 'HL':
                _, _, neurons_dict[l.name] = l.output_shape[0]

                n = int(l.name[2])
                spikes_per_neuron_dict[l.name] = \
                    tf.reduce_sum(output[2*n+1]).numpy() / (batch_size * neurons_dict[l.name])
                weights_per_neuron_dict[l.name] = \
                    tf.reduce_sum(tf.cast(l.weights[0] != 0, dtype='float32')).numpy() / l.weights[0].shape[0]
            elif l.name[0:6] == 'output':
                _, neurons_dict['out'] = l.output_shape

                weights_per_neuron_dict['out'] = \
                    tf.reduce_sum(tf.cast(l.weights[0] != 0, dtype='float32')).numpy() / l.weights[0].shape[0]

        num_synops = np.sum(np.array(list(spikes_per_neuron_dict.values())) *
                            np.array(list(weights_per_neuron_dict.values())) *
                            np.array(nonzero_weight_ratios) *
                            np.array(list(neurons_dict.values()))[:-1])
        num_neurons = np.sum(np.array(list(neurons_dict.values()))[1:])
        return num_synops, num_neurons

    def __call__(self, model, data, nonzero_weight_ratios):
        num_synops, num_neurons = self.calculate_operations(model, data, nonzero_weight_ratios)

        _, timesteps, _ = data.shape

        energy_synops = num_synops * self.energy_per_synops
        energy_neurons = num_neurons * self.energy_per_neuron * timesteps

        energy_total = energy_synops + energy_neurons

        return energy_total


class ModelLatency:
    def __init__(self, device):
        self.device = device
        self.get_latency_values(device)

    def get_latency_values(self, device):
        if device == 'spinnaker':
            self.time_per_neuron = 1.015e-6
            self.time_offset_neuron = 3.235e-6
            self.time_per_synops_f = 0.126e-6
            self.time_offset_synapse_f = 6.567e-6
            self.time_per_synops_s = 0.115e-6
            self.time_offset_synapse_s = 3.96e-6
            self.time_per_synops_l = 0.115e-6
            self.time_offset_synapse_l = 2.48e-6
        else:
            raise Exception(f"Device {self.device} is unknown.")

    @staticmethod
    def calculate_operations(model, data):
        """
        TODO:
        1) consider pruned layer
        2) consider event-triggered updating
        3) current implementation assumes equal connectivity between all neurons
        Also:
        -- calculation is unique to SpiNNaker platform (see SpyNNaker paper)
        """
        output = model(data)

        batch_size, _, in_channels = data.shape

        spikes_per_neuron_dict = dict()
        weights_per_neuron_dict = dict()
        neurons_dict = dict()

        for i, l in enumerate(model.layers):
            if l.name[0:5] == 'input':
                neurons_dict['in'] = in_channels

                spikes_per_neuron_dict['in'] = np.sum(data != 0) / (batch_size * in_channels)
            elif l.name[0:2] == 'HL':
                _, _, neurons_dict[l.name] = l.output_shape[0]

                n = int(l.name[2])
                spikes_per_neuron_dict[l.name] = \
                    tf.reduce_sum(output[2 * n + 1]).numpy() / (batch_size * neurons_dict[l.name])
                weights_per_neuron_dict[l.name] = \
                    tf.reduce_sum(tf.cast(l.weights[0] != 0, dtype='float32')).numpy() / l.weights[0].shape[0]
            elif l.name[0:6] == 'output':
                _, neurons_dict['out'] = l.output_shape

                weights_per_neuron_dict['out'] = \
                    tf.reduce_sum(tf.cast(l.weights[0] != 0, dtype='float32')).numpy() / l.weights[0].shape[0]

        num_layers = len(list(neurons_dict.values())) - 1
        spikes_per_layer_f = np.ones(num_layers)
        spikes_per_layer_s = np.array(list(spikes_per_neuron_dict.values())) * \
                             np.array(list(neurons_dict.values()))[:-1]
        spikes_per_layer_l = np.ones(num_layers)
        weights_per_neuron = np.array(list(weights_per_neuron_dict.values()))
        total_neurons = np.sum(np.array(list(neurons_dict.values()))[1:])

        return spikes_per_layer_f, spikes_per_layer_s, spikes_per_layer_l, weights_per_neuron, total_neurons

    def __call__(self, model, data, nonzero_weight_ratios):
        spikes_per_layer_f, spikes_per_layer_s, spikes_per_layer_l, weights_per_neuron, total_neurons = \
            self.calculate_operations(model, data)

        weights_per_neuron = weights_per_neuron * np.array(nonzero_weight_ratios)

        _, timesteps, _ = data.shape

        time_synops_f = np.sum(spikes_per_layer_f *
                               (self.time_per_synops_f * weights_per_neuron + self.time_offset_synapse_f))
        time_synops_s = np.sum(spikes_per_layer_s *
                               (self.time_per_synops_s * weights_per_neuron + self.time_offset_synapse_s))
        time_synops_l = np.sum(spikes_per_layer_l *
                               (self.time_per_synops_l * weights_per_neuron + self.time_offset_synapse_l))
        time_neurons = total_neurons * self.time_per_neuron * timesteps

        time_total = time_synops_f + time_synops_s + time_synops_l + time_neurons

        return time_total


class SpikeRegularization(tf.keras.losses.Loss):
    def __init__(self, regularization_factor):
        self.regularization_factor = regularization_factor
        super().__init__()

    def call(self, y_true, y_pred):
        spike_share = tf.reduce_sum(y_pred)
        return self.regularization_factor * spike_share

    def get_config(self):
        config = super().get_config()
        config.update({
            'regularization_factor': self.regularization_factor,
        })
        return config
