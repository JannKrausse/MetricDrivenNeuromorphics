import torch.utils.data
import tonic


class Dataset:
    def __init__(self, dataset, train, batchsize, time_bins, spat_ds):
        self.dataset = dataset
        self.train = train
        self.time_bins = time_bins
        self.spat_ds = spat_ds
        self.sensor_size = getattr(tonic.datasets, self.dataset).sensor_size
        self.sensor_size = (int(self.sensor_size[0] * spat_ds), int(self.sensor_size[1] * spat_ds), self.sensor_size[2])
        self.transform = tonic.transforms.Compose(
            [
                tonic.transforms.Downsample(spatial_factor=spat_ds),
                tonic.transforms.ToFrame(sensor_size=self.sensor_size, n_time_bins=self.time_bins),
            ]
        )
        self.savingPath = "/home/kiasic/snailta/Datasets/"
        self.batchsize = batchsize

    def dataloader(self):
        data = getattr(tonic.datasets, self.dataset)(save_to=self.savingPath,
                                      transform=self.transform, train=self.train)
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batchsize,
            num_workers=8,
            persistent_workers=True,
            shuffle=self.train
        )
