import numpy as np


class Metrics:
    def __init__(self, metric_type):
        self.metric_type = metric_type

        self.acc_ref = 0.682
        self.mem_ref = 3294400
        self.energy_ref = 0.035566
        self.latency_ref = 0.52775

    @staticmethod
    def sigmoid(x, a):
        y = 1 / (1 + np.exp(-a * x))
        return y

    def __call__(self, acc, mem, energy, latency):
        if self.metric_type == 'baseline':
            zeta = acc
            return zeta
        elif self.metric_type == 'iot':
            max_memory = 5e4
            zeta = np.heaviside(max_memory - mem, 1) * (np.log10(acc / self.acc_ref) - 100 * (energy - self.energy_ref))
            return zeta
        elif self.metric_type == 'safety':
            min_acc = 0.75
            max_latency = 0.25
            zeta = self.sigmoid(acc - min_acc, 70) * self.sigmoid(max_latency - latency, 100) * \
                    (100 / 5. * (acc - self.acc_ref) - np.log10((mem + 1) / self.mem_ref) -
                     np.log10(energy / self.energy_ref) - np.log10(latency / self.energy_ref))
            return zeta
        elif self.metric_type == 'naive':
            zeta = 100 / 5. * (acc - self.acc_ref) - np.log10((mem + 1) / self.mem_ref) - \
                   np.log10(energy / self.energy_ref) - np.log10(latency / self.energy_ref)
            return zeta
