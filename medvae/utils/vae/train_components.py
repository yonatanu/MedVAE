import os
from time import time
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from rich import print
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
import wandb

from medvae.utils.extras import get_weight_dtype
from medvae.utils.transforms import to_dict

__all__ = ["training_epoch", "validation_epoch"]


def wandb_norm(tensor):
    # Avoid division by zero: if max is zero, use 1.0
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max() if tensor.max() != 0 else 1.0
    return tensor


def training_epoch(
    epoch: int,
    global_step: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    discriminator_iter_start: int,
    default_metrics: List[Metric],
    rec_metrics: List[Metric],
    optimizer_ae: Optimizer,
    optimizer_disc: Optimizer,
    options: Dict[str, Any],
):
    """Train a single epoch of a VAE model."""
    for metric in default_metrics:
        metric.reset()

    for _, metric in rec_metrics:
        metric.reset()

    if len(default_metrics) == 6:
        (
            metric_aeloss,
            metric_discloss,
            metric_ae_recloss,
            metric_bc_loss,
            metric_data,
            metric_batch,
        ) = default_metrics
    else:
        (
            metric_aeloss,
            metric_discloss,
            metric_ae_recloss,
            metric_data,
            metric_batch,
        ) = default_metrics

    model.train()
    epoch_start, batch_start = time(), time()
    dtype = get_weight_dtype(accelerator)
    discloss = torch.tensor(0.0)
    aeloss = torch.tensor(0.0)
    compute_disc = False
    for i, batch in enumerate(dataloader):
        data_time = time() - batch_start

        batch = to_dict(batch)
        if global_step >= discriminator_iter_start:
            compute_disc = True
        images = batch["img"].to(dtype)

        if (compute_disc and i % 2 == 0) or (not compute_disc):
            # Train the encoder and decoder
            reconstructions, posterior, latent = model(images)

            aeloss, log_dict_ae = criterion(
                inputs=images,
                reconstructions=reconstructions,
                posteriors=posterior,
                latent=latent,
                optimizer_idx=0,
                global_step=global_step,
                weight_dtype=dtype,
                last_layer=accelerator.unwrap_model(model).get_last_layer(),
                split="train",
            )

            optimizer_ae.zero_grad()
            accelerator.backward(aeloss)
            optimizer_ae.step()

        elif compute_disc and i % 2 == 1:
            with torch.no_grad():
                reconstructions, posterior, latent = model(images)
            discloss, _log_dict_disc = criterion(
                inputs=images,
                reconstructions=reconstructions,
                posteriors=posterior,
                latent=latent,
                optimizer_idx=1,
                global_step=global_step,
                weight_dtype=dtype,
                last_layer=None,
                split="train",
            )

            optimizer_disc.zero_grad()
            accelerator.backward(discloss)
            optimizer_disc.step()

        # Update metrics
        batch_time = time() - batch_start
        metric_aeloss.update(aeloss)
        metric_ae_recloss.update(log_dict_ae["train/rec_loss"])
        metric_discloss.update(discloss)
        metric_data.update(data_time)
        metric_batch.update(batch_time)

        images, reconstructions = accelerator.gather_for_metrics(
            (images, reconstructions)
        )
        for _, metric in rec_metrics:
            metric.update(images, reconstructions)

        if options.get("is_logging", False) and i == 0:
            input_images = (
                images[:4, 0].detach().cpu() + 1j * images[:4, 1].detach().cpu()
            )
            reconstruction_images = (
                reconstructions[:4, 0].detach().cpu()
                + 1j * reconstructions[:4, 1].detach().cpu()
            )
            input_images = torch.abs(input_images)
            rec_images = torch.abs(reconstruction_images)
            accelerator.log(
                {
                    f"train/input_images": [
                        wandb.Image(wandb_norm(img)) for img in input_images
                    ],
                    f"train/reconstruction_images": [
                        wandb.Image(wandb_norm(img)) for img in rec_images
                    ],
                }
            )

        # Logging values
        print(
            f"\r[Epoch <{epoch:03}/{options['max_epoch']}>: Step <{i:03}/{len(dataloader)}>] - "
            + f"Data(s): {data_time:.3f} ({metric_data.compute():.3f}) - "
            + f"Batch(s): {batch_time:.3f} ({metric_batch.compute():.3f}) - "
            + f"AE Loss: {aeloss.item():.3f} ({metric_aeloss.compute():.3f}) - "
            + f"AE Rec Loss: {log_dict_ae['train/rec_loss'].item():.3f} ({metric_ae_recloss.compute():.3f}) - "
            + f"Disc Loss: {discloss.item():.3f} ({metric_discloss.compute():.3f}) - "
            + f"{(((time() - epoch_start) / (i + 1)) * (len(dataloader) - i)) / 60:.2f} m remaining\n"
        )

        if options["is_logging"] and i % options["log_every_n_steps"] == 0:
            log_data = {
                "epoch": epoch,
                "mean_aeloss": metric_aeloss.compute(),
                "mean_ae_recloss": metric_ae_recloss.compute(),
                "mean_discloss": metric_discloss.compute(),
                "mean_data": metric_data.compute(),
                "mean_batch": metric_batch.compute(),
                "step": i,
                "step_global": global_step,
                "step_aeloss": aeloss,
                "step_ae_recloss": log_dict_ae["train/rec_loss"],
                "step_discloss": discloss,
                "step_data": data_time,
                "step_batch": batch_time,
            }

            for name, metric in rec_metrics:
                log_data[name] = metric.compute()

            accelerator.log(log_data)
        global_step += 1

        if global_step % options["ckpt_every_n_steps"] == 0:
            try:
                accelerator.save_state(
                    os.path.join(options["ckpt_dir"], f"step_{global_step}.pt")
                )
            except Exception as e:
                print(e)

        batch_start = time()

        if options["fast_dev_run"]:
            break

    return global_step


def validation_epoch(
    options: Dict[str, Any],
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    default_metrics: List[Metric],
    rec_metrics: List[Metric],
    global_step: int = 0,
    postfix: str = "",
):
    """Validate one epoch for the VAE model."""
    for metric in default_metrics:
        metric.reset()

    for _, metric in rec_metrics:
        metric.reset()

    metric_aeloss = default_metrics[0]
    metric_discloss = default_metrics[1]
    metric_ae_recloss = default_metrics[2]
    metric_aeloss.reset()
    metric_discloss.reset()
    metric_ae_recloss.reset()

    model.eval()
    criterion.eval()
    epoch_start = time()
    dtype = get_weight_dtype(accelerator)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = to_dict(batch)

            images = batch["img"].to(dtype)
            reconstructions, posterior, latent = model(images)

            aeloss, log_dict_ae = criterion(
                inputs=images,
                reconstructions=reconstructions,
                posteriors=posterior,
                latent=latent,
                optimizer_idx=0,
                global_step=global_step,
                weight_dtype=dtype,
                last_layer=accelerator.unwrap_model(model).get_last_layer(),
                split="valid",
            )

            discloss, _log_dict_disc = criterion(
                inputs=images,
                reconstructions=reconstructions,
                posteriors=posterior,
                latent=latent,
                optimizer_idx=1,
                global_step=global_step,
                weight_dtype=dtype,
                last_layer=accelerator.unwrap_model(model).get_last_layer(),
                split="valid",
            )

            metric_aeloss.update(aeloss)
            metric_ae_recloss.update(log_dict_ae["valid/rec_loss"])
            metric_discloss.update(discloss)
            images, reconstructions = accelerator.gather_for_metrics(
                (images, reconstructions)
            )
            for _, metric in rec_metrics:
                metric.update(images, reconstructions)

            # Logging values
            print(
                f"\r Validation{postfix}: "
                + f"\r[Epoch <{epoch:03}/{options['max_epoch']}>: Step <{i:03}/{len(dataloader)}>] - "
                + f"AE Loss: {aeloss.item():.3f} ({metric_aeloss.compute():.3f}) - "
                + f"AE Rec Loss: {log_dict_ae['valid/rec_loss'].item():.3f} ({metric_ae_recloss.compute():.3f}) - "
                + f"Disc Loss: {discloss.item():.3f} ({metric_discloss.compute():.3f}) - "
                + f"{(((time() - epoch_start) / (i + 1)) * (len(dataloader) - i)) / 60:.2f} m remaining\n"
            )

            if options["fast_dev_run"]:
                break

            if options.get("is_logging", False) and i == 0:
                input_images = (
                    images[:4, 0].detach().cpu() + 1j * images[:4, 1].detach().cpu()
                )
                reconstruction_images = (
                    reconstructions[:4, 0].detach().cpu()
                    + 1j * reconstructions[:4, 1].detach().cpu()
                )
                input_images = torch.abs(input_images)
                rec_images = torch.abs(reconstruction_images)
                accelerator.log(
                    {
                        f"valid/input_images": [
                            wandb.Image(wandb_norm(img)) for img in input_images
                        ],
                        f"valid/reconstruction_images": [
                            wandb.Image(wandb_norm(img)) for img in rec_images
                        ],
                    }
                )

        if options["is_logging"]:
            log_data = {
                f"valid{postfix}/epoch": epoch,
                f"valid{postfix}/mean_aeloss": metric_aeloss.compute(),
                f"valid{postfix}/mean_ae_recloss": metric_ae_recloss.compute(),
                f"valid{postfix}/mean_discloss": metric_discloss.compute(),
            }
            for name, metric in rec_metrics:
                log_data[f"valid{postfix}/{name}"] = metric.compute()

            accelerator.log(log_data)

    return metric_ae_recloss.compute()
