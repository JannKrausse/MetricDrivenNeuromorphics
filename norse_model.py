"""This code is taken from the Norse library (https://github.com/norse/norse) due to problems when importing it as a
whole."""
import torch, torch.nn as nn
import pytorch_lightning as pl
import torch.optim as to
import torchmetrics as tm

from norse_neuron import LIF, Integrator, LIFParameters, IntegratorParameters


class ShallowSpiker(pl.LightningModule):
    def __init__(self, n_in, n_hidden, n_out, tau, lr=1e-3):
        """Initialize model generically."""
        super().__init__()
        self.tau = tau
        self.lr = lr

        # projection layers
        self.flatten1 = nn.Flatten(start_dim=2)  # only flatten x, y, and polarity

        # linear layers
        self.linear1 = nn.Linear(n_in, n_hidden, bias=False, dtype=torch.float32)
        self.linear2 = nn.Linear(n_hidden, n_out, bias=False, dtype=torch.float32)
        self.linear_test = nn.Linear(n_in, n_out, bias=False, dtype=torch.float32)

        # spiking layers
        self.lif1 = LIF(p=LIFParameters(), dt=1.0)

        # integrator layer
        self.li1 = Integrator(p=IntegratorParameters(), dt=1.0)

    def forward(self, batch):
        """define forward pass for inference"""
        x, y = batch
        x = x.type(torch.float32)

        # Tonic dataset specific:
        # transpose the 1st and 2nd dimension, as torch expects shape (seq_length, batch_size, ...)
        x = x.transpose(0, 1)

        z = self.flatten1(x)
        z = self.linear1(z)
        z1, _ = self.lif1(z)
        z2 = self.linear2(z1)
        out, _ = self.li1(z2)

        return out, y, z1

    def decode_max(self, out):
        out, _ = out.max(dim=0)
        return out

    def decode_last(self, out):
        out = out[-1]
        return out

    def training_step(self, batch, batch_idx):
        out, y = self.forward(batch)

        # decode output
        out = self.decode_last(out)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # select gpu

        # calculate and log loss
        loss = torch.nn.functional.cross_entropy(out, y)
        self.log("train_loss", loss, prog_bar=True, logger=False, on_step=True,
                 on_epoch=True)

        # calculate and log accuracy
        accuracy = tm.Accuracy(task="multiclass", num_classes=out.size()[1]).to(device)(out, y)
        # accuracy = tm.Accuracy(task="multiclass", num_classes=out.size()[1])(out, y)
        self.log("train_acc", accuracy, prog_bar=True, logger=False, on_step=True,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out, y = self.forward(batch)

        out, _ = out.max(dim=0)

        device = 'cuda' if torch.cuda.is_available() else 'cpu' # select gpu

        # calculate and log loss
        loss = torch.nn.functional.cross_entropy(out, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # calculate and log accuracy
        accuracy = tm.Accuracy(task="multiclass", num_classes=out.size()[1]).to(device)(out, y)
        self.log("val_acc", accuracy, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return to.Adam(self.parameters(), lr=self.lr)
