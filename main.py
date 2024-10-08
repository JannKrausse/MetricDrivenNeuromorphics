import datetime
from pathlib import Path
import argparse
from distutils.util import strtobool

from data import get_data_tonic
from model import create_model, compile_model, save_model
from train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='DVSGesture', type=str)
    parser.add_argument('--nb_bins', default=100, type=int)
    parser.add_argument('--spat_ds_fac', default=0.25, type=float)
    parser.add_argument('--neuron_type', default='Std', type=str)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_hidden_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--batch_size', default=1100, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--es_patience', default=50, type=int)
    parser.add_argument('--custom_name', default="", type=str)
    parser.add_argument('--recurrence', dest='recurrence', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--fixed', dest='fixed', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--eagerly', dest='eagerly', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--leaky_output', dest='leaky_output', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--spike_regu', default=1.7615947321726526e-06, type=float)
    parser.add_argument('--weight_regu', default=1.1942128822739302e-07, type=float)
    args = parser.parse_args()
    
    from gpu_selection import select_gpu
    select_gpu(0)
    import tensorflow as tf

    dataset = get_data_tonic(dataset_name=args.dataset_name, nb_bins=args.nb_bins, spat_ds_fac=args.spat_ds_fac)
    in_shape = (dataset['x_train_set'][0].shape[0], dataset['x_train_set'][0].shape[1])
    out_shape = tf.unique(dataset['y_valid_set']).y.shape[0]

    model = create_model(neuron_type=args.neuron_type, input_shape=in_shape,
                         hidden_units=args.hidden_units, num_hidden_layers=args.num_hidden_layers,
                         output_shape=out_shape, fixed=args.fixed, dropout=args.dropout,
                         recurrence=args.recurrence, leaky_output=args.leaky_output, weight_regu=args.weight_regu)

    model = compile_model(model, lr=args.lr, eagerly=args.eagerly, batch_size=args.batch_size,
                          spike_regu_factor=args.spike_regu)

    model = train_model(model, dataset, batch_size=args.batch_size, epochs=args.epochs, es_patience=args.es_patience)

    from model_conversion import ConvertedTorchModel
    from torch_data import Dataset
    import torch
    from auxiliary import ModelLatency, ModelEnergy
    model_energy = ModelEnergy(device='spinnaker')
    model_latency = ModelLatency(device='spinnaker')

    bit_length = [None, 24, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]

    accs_array = []
    memory_array = []
    energy_array = []
    latency_array = []
    for bl in bit_length:
        with torch.no_grad():
            converted_model = ConvertedTorchModel(model, bit_length=bl, quantization_type='fixed_point')
            torch_model = converted_model.torch_model

            torch_dataset = Dataset(dataset=args.dataset_name, train=False, batchsize=args.batch_size,
                                    time_bins=args.nb_bins, spat_ds=args.spat_ds_fac).dataloader()
            torch_data = next(iter(torch_dataset))
            acc, _lif_output, _integrator_output = converted_model.get_torch_model_acc(torch_data)
            accs_array.append(acc)

            def get_nonzero_weights(torch_model):
                nonzero_weights = 0
                nonzero_weight_ratios = []
                for l in [torch_model.linear1, torch_model.linear2]:
                    w = l.weight
                    nonzero_weights += torch.sum(w != 0.)
                    shape = l.weight.shape[0] * l.weight.shape[1]
                    nonzero_weight_ratios.append(torch.sum(w != 0.) / shape)

                return nonzero_weights, nonzero_weight_ratios

            nonzero_weights, nonzero_weight_ratios = get_nonzero_weights(torch_model)
            if bl is None:
                bl = 32
            memory_array.append(nonzero_weights.numpy() * bl)

            energy_array.append(model_energy(model, dataset['x_valid_set'], nonzero_weight_ratios))
            latency_array.append(model_latency(model, dataset['x_valid_set'], nonzero_weight_ratios))

    bit_length[0] = 32
    import matplotlib.pyplot as plt
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlabel('bitwidth')
    ax1.set_ylabel('acc')
    ax1.plot(bit_length, accs_array, label='', c='green')
    ax1.legend(loc='best')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel('bitwidth')
    ax2.set_ylabel('memory')
    ax2.plot(bit_length, memory_array, label='', c='blue')
    ax2.legend(loc='best')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlabel("bitwidth")
    ax3.set_ylabel("energy")
    ax3.plot(bit_length, energy_array, label='', c='orange')
    ax3.legend(loc='best')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlabel('bitwidth')
    ax4.set_ylabel('latency')
    ax4.plot(bit_length, latency_array, label='')
    ax4.legend(loc='best')

    plt.show()
    plt.savefig("results_plot")

    time_now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
    custom_name = args.custom_name
    save_model_path = Path(__file__).parent / f"saved_models/{args.dataset_name}_{time_now}{custom_name}"
    save_model(model, saving_path=save_model_path)

    print("Script execution finished")


if __name__ ==  "__main__":
    main()
