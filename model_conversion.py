from norse_model import ShallowSpiker
import torch
import torch.nn as nn
import numpy as np
from qtorch.quant import Quantizer
from qtorch import FixedPoint, FloatingPoint
import tensorflow as tf
import math


class ConvertedTorchModel:
    """This class takes a TF model and creates a Torch model with the same architecture and same weights.
    For this, it assumes two things:
        - it is a LIF-based network
        - it is a non-recurrent feed-forward network
    """
    def __init__(self, tf_model, bit_length, quantization_type='fixed_point'):
        self.tf_model = tf_model
        self.tau = tf_model.layers[2].cell.tau
        self.bit_length = bit_length

        # agents for weight quantization
        if self.bit_length is None:
            self.weight_quantizer = torch.nn.Identity()
        else:
            if quantization_type == 'fixed_point':
                w_max = tf.reduce_max(tf.abs(tf_model.layers[2].weights[0])).numpy()
                word_length = int(np.log2(w_max))
                if word_length > bit_length:
                    raise Exception('This does not work! Fixed point quantization does not work for these values.')
                self.weight_quantizer = Quantizer(FixedPoint(wl=self.bit_length, fl=self.bit_length - word_length - 1),
                                                  forward_rounding="nearest")
            elif quantization_type == 'floating_point':
                exp = math.ceil(self.bit_length / 4.)
                man = self.bit_length - exp
                if man > 23:
                    man = 23
                    print("Attention: mantissa is shortened to 23 bits!")
                self.weight_quantizer = Quantizer(FloatingPoint(man=man, exp=exp), forward_rounding="nearest")
            else:
                raise Exception(f'Quantization type {quantization_type} is unknown.')

        # extract architecture from tensorflow model
        self.num_layers = 0
        self.layer_units = dict()
        self.weights = dict()
        for l in tf_model.layers:
            if l.name[0:2] == 'in':
                _, _, self.layer_units['in'] = l.input_shape[0]
            elif l.name[0:2] == 'HL':
                self.num_layers += 1
                if self.num_layers > 1:
                    raise Exception('WATCH OUT! The ShallowSpiker class currently does not support more than one '
                                    'hidden layer!')
                _, _, self.layer_units[f'{l.name[2]}'] = l.output_shape[0]
                self.weights[f'{l.name[2]}'] = l.weights[0].numpy()
            elif l.name[0:3] == 'out':
                _, self.layer_units['out'] = l.output_shape
                self.weights['out'] = l.weights[0].numpy()

        # create torch model with same architecture as tensorflow model
        self.torch_model = ShallowSpiker(*self.layer_units.values(), tau=self.tau)

        # assign same weights to norse model as tensorflow model
        for k, w in self.weights.items():
            if k[0] != 'o':
                getattr(self.torch_model, f'linear{int(k)+1}').weight = \
                    nn.Parameter(self.weight_quantizer(torch.tensor(w.transpose())))
            else:
                getattr(self.torch_model, f'linear{self.num_layers+1}').weight = \
                    nn.Parameter(self.weight_quantizer(torch.tensor(w.transpose())))

    def get_torch_model_acc(self, torch_data):
        with torch.no_grad():
            torch_output = self.torch_model(torch_data)
            torch_output_last_ts = torch_output[0][-1]
            output_prediction = torch.argmax(torch_output_last_ts, dim=1)
            true_predictions = (output_prediction == torch_output[1]).numpy()
            acc = np.sum(true_predictions) / true_predictions.size

        lif_output = torch_output[2]
        integrator_output = torch_output_last_ts

        return acc, lif_output, integrator_output
