import argparse
from datasets.balls import (
    SparseBall,
    BlockOffset,
)
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from torchvision import models as vision_models
from torchvision import transforms
import os
import wandb

from disentanglement_utils import linear_disentanglement, permutation_disentanglement

if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# constants used for normalizing the data
MU = np.array([0.9906, 0.9902, 0.9922])
SIG = np.array([0.008, 0.008, 0.008])


def setup_parser():
    parser = argparse.ArgumentParser(description="Sparse Action Disentanglement")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_offsets", default=1, type=int)
    parser.add_argument("--n_balls", default=1, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument(
        "--workers", default=0, type=int, help="Number of workers to use (0=#cpus)"
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--true_mech", action="store_true")
    parser.add_argument("--combinations", action="store_true")
    parser.add_argument("--block_offset", action="store_true")
    return parser


class Encoder(pl.LightningModule):
    def __init__(self, n_latents: int, args, base_architecture="resnet18", width=128):
        super().__init__()
        base_model = getattr(vision_models, base_architecture)
        layers = [
            base_model(False, num_classes=width),
            nn.LeakyReLU(),
            nn.Linear(width, width),
            nn.LeakyReLU(),
            nn.Linear(width, n_latents),
        ]
        model = nn.Sequential(*layers)
        self.encoder = torch.nn.DataParallel(model)
        self.lr = args.lr
        self.n_latents = n_latents
        self._load_messages()

    def _load_messages(self):
        pass

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, valid_batch, batch_idx):
        (z1, _), (x1, _), _ = valid_batch

        pred_z = self(x1)
        return {"true_z": z1, "pred_z": pred_z}

    def validation_epoch_end(self, validation_step_outputs):
        z_disentanglement = [v["true_z"] for v in validation_step_outputs]
        h_z_disentanglement = [v["pred_z"] for v in validation_step_outputs]
        z_disentanglement = torch.cat(z_disentanglement, 0)
        h_z_disentanglement = torch.cat(h_z_disentanglement, 0)
        (linear_disentanglement_score, _), _ = linear_disentanglement(
            z_disentanglement, h_z_disentanglement, mode="r2", train_test_split=True
        )

        (permutation_disentanglement_score, _), _ = permutation_disentanglement(
            z_disentanglement,
            h_z_disentanglement,
            mode="pearson",
            solver="munkres",
            rescaling=True,
        )
        mse = F.mse_loss(z_disentanglement, h_z_disentanglement).mean(0)
        self.log("Linear_Disentanglement", linear_disentanglement_score, prog_bar=True)
        self.log(
            "Permutation Disentanglement",
            permutation_disentanglement_score,
            prog_bar=True,
        )
        self.log("MSE", mse, prog_bar=True)
        wandb.log(
            {
                "mse": mse,
                "Permutation Disentanglement": permutation_disentanglement_score,
                "Linear_Disentanglement": linear_disentanglement_score,
            }
        )


class Ident(Encoder):
    def __init__(self, n_latents, args):
        super().__init__(n_latents=n_latents, args=args)

    def _load_messages(self):
        print(f"Individual offsets model")

    def loss(self, m_z1, z2):
        return F.mse_loss(m_z1, z2)

    def training_step(self, train_batch, batch_idx):
        (_, _), (x1, x2), (_, b) = train_batch
        z1 = self(x1).squeeze()
        m_z1 = z1 + b.squeeze()
        z2 = self(x2).squeeze()
        loss = self.loss(m_z1, z2)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()
        self.log("train_loss", loss)
        wandb.log({"loss": loss})
        return loss


class IdentBlock(Encoder):
    def __init__(self, n_latents, args):
        super().__init__(n_latents=n_latents, args=args)
        init = torch.rand(args.n_balls * 2) * 0.1 + 0.5
        self.offset = nn.parameter.Parameter(init, requires_grad=True)
        self.true_mech = args.true_mech

    def loss(self, m_z1, z2):
        return F.mse_loss(m_z1, z2)

    def training_step(self, train_batch, batch_idx):
        (_, _), (x1, x2), (_, b) = train_batch
        # clamp the learnt offset to be nonzero
        with torch.no_grad():
            self.offset[self.offset.abs() < 0.2] = (
                torch.sign(self.offset)[self.offset.abs() < 0.2] * 0.2
            )
        z1 = self(x1).squeeze()
        if self.true_mech:
            m_z1 = z1 + b
        else:
            mask = b.squeeze().abs() > 0
            m_z1 = z1 + self.offset[None, :] * mask
        z2 = self(x2)
        loss = self.loss(m_z1, z2)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()
        self.log("train_loss", loss)
        wandb.log({"loss": loss})
        return loss


def main(args, name="test-run", project=""):
    pl.utilities.seed.seed_everything(args.seed)

    print("Setting up transformations")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MU,  # channel means - the images are mostly white so close to 1.
                std=SIG,
            ),
        ]
    )
    if args.block_offset:
        print("Using Block offset datset")
        train_dataset = BlockOffset(
            transform=transform,
            n_balls=args.n_balls,
            true_mech=args.true_mech,
            combination_offsets=args.combinations,
        )
    else:
        train_dataset = SparseBall(
            transform=transform,
            n_balls=args.n_balls,
            true_mech=args.true_mech,
            n_offsets=args.n_offsets,
        )
    # data
    print("Setting up data loaders")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # validation dataset is the same as thee train dataset because
    # examples are constructed online.
    validation_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # model
    print("Setting up model")
    if args.block_offset:
        print("Using Block offset model")
        model = IdentBlock(2 * args.n_balls, args)
    else:
        model = Ident(2 * args.n_balls, args)
    wandb.watch(
        model,
        criterion=None,
        log="gradients",
        log_freq=1000,
        idx=None,
        log_graph=(False),
    )

    # training
    print("Starting training")

    trainer = pl.Trainer(
        gpus=1,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=5,
        max_epochs=1_000_000,
        callbacks=[],
    )
    trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    project = "ball-inertia-sparse"
    parser = setup_parser()
    args = parser.parse_args()
    wandb.init(project=project, config={})
    wandb.config.update(args)
    wandb.run.log_code(".")
    main(args, name=wandb.run.name, project=project)
