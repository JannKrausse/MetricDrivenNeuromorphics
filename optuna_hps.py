import os
from pathlib import Path
import argparse
import joblib
from distutils.util import strtobool

from gpu_selection import select_gpu
select_gpu(0)
import tensorflow as tf
import numpy as np
import optuna
import matplotlib.pyplot as plt
import torch

from data import get_data_tonic
from train import train_model
from model import create_model, compile_model
from auxiliary import ModelLatency, ModelEnergy
from model_conversion import ConvertedTorchModel
from torch_data import Dataset
from metrics import Metrics


def save_study_results(study, study_name):
    current_dir = Path(__file__).parent
    save_study_dir_name = f"optuna_results/{study_name}"
    save_study_path = current_dir / save_study_dir_name

    if not os.path.isdir(save_study_path):
        os.mkdir(save_study_path)

    if not os.path.isfile(f"{save_study_path}/results.txt"):
        np.savetxt(f"{save_study_path}/results.txt", [])

    results = np.loadtxt(f"{save_study_path}/results.txt")

    results = np.append(results, f"Best trial number: {study.best_trial.number}")
    results = np.append(results, f" Acc: {study.best_trial.value}")
    for k, v in study.best_params.items():
        results = np.append(results, f"  {k}: {v}")
    results = np.append(results, "------------------")
    for i, t in enumerate(study.trials):
        results = np.append(results, f"Trial number: {i}")
        results = np.append(results, f" Acc: {t.value}")
        for k, v in t.params.items():
            results = np.append(results, f"  {k}: {v}")

    np.savetxt(f"{save_study_path}/results.txt", results, fmt='%s')


def plot_study_results(study, study_name):
    current_dir = Path(__file__).parent
    save_study_dir_name = f"optuna_results/{study_name}"
    save_study_path = current_dir / save_study_dir_name

    param_values_dict = dict()
    acc_values = np.array([])
    l = len(study.trials[0].params.keys())
    for i in range(l):
        param_values_dict[i] = np.array([])
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            for i, v in enumerate(t.params.values()):
                param_values_dict[i] = np.append(param_values_dict[i], v)
            acc_values = np.append(acc_values, t.value)

    np.savetxt(save_study_path/"_acc", acc_values)
    for i in range(l):
        np.savetxt(save_study_path / f"_{list(study.trials[0].params.keys())[i]}", param_values_dict[i])

    for i in range(l):
        plt.plot(param_values_dict[i], acc_values, ls='', marker='.')
        plt.xlabel(f'{list(study.trials[0].params.keys())[i]}')
        plt.ylabel(f'accuracy')
        plt.xscale('log')
        plt.savefig(save_study_path/f"results_plot_{list(study.trials[0].params.keys())[i]}")
        plt.clf()


def optuna_main(dataset, torch_data, spike_regu, weight_regu, lr, batch_size, es_patience, bit_lengths,
                train_args=None):

    if train_args is None:
        train_args = dict()

    callbacks = []

    in_shape = (dataset['x_train_set'][0].shape[0], dataset['x_train_set'][0].shape[1])
    out_shape = tf.unique(dataset['y_valid_set']).y.shape[0]

    train_args['fixed'] = False
    train_args['input_shape'] = in_shape
    train_args['output_shape'] = out_shape
    train_args['dropout'] = 0.4

    model = create_model(**train_args, weight_regu=weight_regu)

    model = compile_model(model, lr=lr, eagerly=False,
                          spike_regu_factor=spike_regu, batch_size=batch_size)

    model = train_model(model, dataset, batch_size=batch_size, epochs=10000, es_patience=es_patience,
                        callbacks=callbacks)

    model_energy = ModelEnergy(device='spinnaker')
    model_latency = ModelLatency(device='spinnaker')

    accs_array = []
    memory_array = []
    energy_array = []
    latency_array = []
    for bl in bit_lengths:
        with torch.no_grad():
            converted_model = ConvertedTorchModel(model, bit_length=bl, quantization_type='fixed_point')
            torch_model = converted_model.torch_model

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
            memory_array.append(nonzero_weights * bl)

            energy_array.append(model_energy(model, dataset['x_valid_set'], nonzero_weight_ratios))
            latency_array.append(model_latency(model, dataset['x_valid_set'], nonzero_weight_ratios))

    return np.array(accs_array), np.array(memory_array), np.array(energy_array), np.array(latency_array)


class Objective:
    def __init__(self, dataset, torch_dataset, num_models, batch_size, lr, es_patience, train_args, bit_lengths,
                 metric_type):
        self.dataset = dataset
        self.torch_dataset = torch_dataset
        self.num_models = num_models
        self.batch_size = batch_size
        self.lr = lr
        self.es_patience = es_patience
        self.train_args = train_args
        self.bit_lengths = bit_lengths
        self.metric_type = metric_type

    def __call__(self, trial):
        spike_regu = trial.suggest_float('spike_regu', 1e-10, 1, log=True)
        weight_regu = trial.suggest_float('weight_regu', 1e-10, 1, log=True)

        best_metric = []
        acc_best_bitlength = []
        memory_best_bitlength = []
        energy_best_bitlength = []
        latency_best_bitlength = []
        best_bitlength = []
        for _ in range(self.num_models):
            arrays = optuna_main(dataset=self.dataset, spike_regu=spike_regu, weight_regu=weight_regu,
                                batch_size=self.batch_size, lr=self.lr, es_patience=self.es_patience,
                                bit_lengths=self.bit_lengths, train_args=self.train_args,
                                torch_data=self.torch_dataset)
            accs_array, memory_array, energy_array, latency_array = arrays
            self.bit_lengths[0] = 32

            metric = Metrics(self.metric_type)(accs_array, memory_array, energy_array, latency_array)

            max_idx = np.argmax(metric)
            best_metric.append(metric[max_idx])
            acc_best_bitlength.append(accs_array[max_idx])
            memory_best_bitlength.append(memory_array[max_idx])
            energy_best_bitlength.append(energy_array[max_idx])
            latency_best_bitlength.append(latency_array[max_idx])
            best_bitlength.append(self.bit_lengths[max_idx])

        trial.set_user_attr('acc_best_bitwidth', np.array(acc_best_bitlength).mean())
        trial.set_user_attr('memory_best_bitwidth', np.array(memory_best_bitlength).mean())
        trial.set_user_attr('energy_best_bitwidth', np.array(energy_best_bitlength).mean())
        trial.set_user_attr('latency_best_bitwidth', np.array(latency_best_bitlength).mean())
        trial.set_user_attr('best_bitwidth', np.array(best_bitlength).mean())

        return np.array(best_metric).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='DVSGesture', type=str)
    parser.add_argument('--spat_ds', default=0.25, type=float)
    parser.add_argument('--nb_time_bins', default=100, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--batch_size', default=1100, type=int)
    parser.add_argument('--nb_hidden_layers', default=1, type=int)
    parser.add_argument('--neuron_type', default='Std', type=str)
    parser.add_argument('--recurrence', dest='recurrence', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--leaky_output', dest='leaky_output', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--load_existing_study', dest='load_existing_study', type=lambda x: bool(strtobool(x)),
                        default=False)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--es_patience', default=50, type=int)
    parser.add_argument('--nb_models_per_trial', default=1, type=int)
    parser.add_argument('--nb_trials', default=50, type=int)
    parser.add_argument('--custom_name', default='', type=str)
    parser.add_argument('--bit_lengths', default=[32, 24, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2], type=list[int])
    parser.add_argument('--metric_type', default='naive', type=str)
    args = parser.parse_args()

    train_args = dict()

    dataset_name = args.dataset_name

    train_args['dropout'] = 0.4
    train_args['recurrence'] = args.recurrence
    train_args['leaky_output'] = args.leaky_output
    train_args['neuron_type'] = args.neuron_type
    train_args['hidden_units'] = args.hidden_units
    train_args['num_hidden_layers'] = args.nb_hidden_layers
    nb_bins = args.nb_time_bins

    dataset = get_data_tonic(dataset_name=dataset_name, nb_bins=nb_bins, spat_ds_fac=args.spat_ds)
    torch_dataset = Dataset(dataset=args.dataset_name, train=False, batchsize=args.batch_size,
                            time_bins=args.nb_time_bins, spat_ds=args.spat_ds)
    torch_dataset_loader = torch_dataset.dataloader()
    torch_data = next(iter(torch_dataset_loader))

    study_name = f"{args.metric_type}_{dataset_name}-{nb_bins}-{args.spat_ds}_{train_args['neuron_type']}_" \
                 f"recurrence={args.recurrence}" \
                 f"_leakyoutput={args.leaky_output}_{train_args['hidden_units']}neurons_" \
                 f"{int(100*train_args['dropout'])}dropout_{args.nb_models_per_trial}modelsmean_{args.custom_name}"

    load_existing_study = args.load_existing_study
    n_trials_per_cycle = 1
    n_cycles = int(args.nb_trials / n_trials_per_cycle)  # total number of trials is n_trials_per_save*n_cycles
    parent = Path(__file__).parent
    if not os.path.isdir(parent / 'CurrentOptunaStudyDump'):
        os.mkdir(parent / 'CurrentOptunaStudyDump')
    if not load_existing_study:
        lg = 0
    else:
        lg = len(joblib.load(parent / f'CurrentOptunaStudyDump/{study_name}.pkl').trials)
    for i in range(n_cycles - int(lg / n_trials_per_cycle)):
        if i == 0 and load_existing_study is False:
            study = optuna.create_study(direction='maximize')  # , pruner=optuna.pruners.HyperbandPruner())
        else:
            study = joblib.load(parent / f'CurrentOptunaStudyDump/{study_name}.pkl')
        objective = Objective(num_models=args.nb_models_per_trial, dataset=dataset, train_args=train_args, lr=args.lr,
                              batch_size=args.batch_size, es_patience=args.es_patience,
                              bit_lengths=args.bit_lengths, torch_dataset=torch_data, metric_type=args.metric_type)
        study.optimize(objective, n_trials=n_trials_per_cycle, gc_after_trial=True)
        joblib.dump(study, parent / f'CurrentOptunaStudyDump/{study_name}.pkl')

    save_study_results(study, study_name)
    plot_study_results(study, study_name)


if __name__ == "__main__":
    main()
