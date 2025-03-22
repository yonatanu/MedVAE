import torch
import argparse
import pyrootutils
from genericpath import isdir
from medvae.utils.extras import create_directory, cite_function
from medvae.utils.factory import create_model
from medvae.utils.extras import sanitize_dataloader_kwargs, set_seed
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import GradientAccumulationPlugin
from lora_diffusion import inject_trainable_lora_extended
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from medvae.utils.vae.train_components import training_epoch, validation_epoch

"""
Process configuration from Hydra instead of command line arguments.

Args:
    cfg: Hydra configuration object
    
Returns:
    The processed configuration: hydra config object
"""
def parse_arguments(cfg: DictConfig):

    # Print a message to cite the med-vae paper
    cite_function()
    
    # Check if input folder exists
    assert isdir(cfg.input), f"Input folder {cfg.input} does not exist."
    
    # Validate model_name
    valid_model_names = ['medvae_4_1_2d', 'medvae_4_3_2d', 'medvae_4_4_2d', 'medvae_8_1_2d', 'medvae_8_4_2d', 'medvae_4_1_3d', 'medvae_8_1_3d']
    assert cfg.model_name in valid_model_names, f"model_name must be one of {valid_model_names}. Got: {cfg.model_name}."
    
    assert cfg.stage2 in [True, False], f"stage2 must be one of [True, False]. This is used for 2D stage 2 finetuning. Got: {cfg.stage2}."
    
    if cfg.stage2 == 'true':
        assert os.path.exists(cfg.stage2_weight), f"stage2_weight {cfg.stage2_weight} does not exist."
    
    return cfg

# Set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")

# Register configuration resolvers
OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base="1.2", config_path=config_dir, config_name="finetuned_vae.yaml")
def main(cfg: DictConfig):
    cfg = parse_arguments(cfg)
        
    # Instantiating config
    print(f"=> Starting [experiment={cfg.get('task_name', 'default')}]")
    cfg = instantiate(cfg)
    
    # Seeding
    if cfg.get("seed", None) is not None:
        print(f"=> Setting seed [seed={cfg.seed}]")
        set_seed(cfg.seed)
        
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Setup accelerator
    logger_kwargs = cfg.get("logger", None)
    is_logging = bool(logger_kwargs)
    print(f"=> Instantiate accelerator [logging={is_logging}]")

    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            adjust_scheduler=False,
        ),
        mixed_precision=cfg.get("mixed_precision", None),
        log_with="wandb" if is_logging else None,
        split_batches=True,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True,
            )
        ],
    )
    accelerator.init_trackers("compress", config=cfg, init_kwargs={"wandb": logger_kwargs})

    # Determine the mode
    print(f"=> Mixed precision: {accelerator.mixed_precision}")

    inference_mode = cfg.get("inference", False)
    print(f"=> Running in inference mode: {inference_mode}")

    print(f"=> Instantiating train dataloader [device={accelerator.device}]")
    train_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["train"]))
    
    print(f"=> Instantiating valid dataloader [device={accelerator.device}]")
    valid_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["valid"]))

    # Create loss function
    criterion = cfg.criterion
    discriminator_iter_start = criterion.discriminator_iter_start

    # Create model and use prior weight for stage 2 finetuning
    if cfg.stage2 == 'true':
        model = create_model(cfg.model_name, existing_weight=cfg.stage2_weight)
    else:
        model = create_model(cfg.model_name)
        
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Create two optimizers: one for the autoencoder and one for the discriminator
    print(f"=> Instantiating the optimizer [device={accelerator.device}]")

    batch_size, lr = cfg.batch_size, cfg.base_learning_rate
    lr = gradient_accumulation_steps * batch_size * lr
    
    # Create autoencoder parameters
    ae_params = (
    list(model.encoder.parameters())
    + list(model.decoder.parameters())
    + list(model.quant_conv.parameters())
    + list(model.post_quant_conv.parameters())
    )
    
    if criterion.learn_logvar:
        ae_params.append(criterion.logvar)
    opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(criterion.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    num_metrics = 5
    # This mean uses lora and needs biomedclip loss
    if cfg.model_name in ['medvae_4_3_2d', 'medvae_8_4_2d']:
        num_metrics += 1   
    
    # Prepare components for multi-gpu/mixed precision training
    (train_dataloader, valid_dataloader, model, opt_ae, opt_disc, criterion) = accelerator.prepare(
        train_dataloader,
        valid_dataloader,
        model,
        opt_ae,
        opt_disc,
        criterion,
    )
    
    # Create metrics: aeloss, discloss, recloss, data(time), batch(time)
    default_metrics = accelerator.prepare(*[MeanMetric() for _ in range(num_metrics)])
    if len(cfg["metrics"]) > 0:
        names, metrics = list(zip(*cfg["metrics"]))
        metrics = list(zip(names, accelerator.prepare(*metrics)))
    else:
        metrics = []
        
    # Resume from checkpoint
    start_epoch = cfg.start_epoch
    if cfg.resume_from_ckpt is not None:
        print('Loading Model from Checkpoint: ', cfg.resume_from_ckpt)
        accelerator.load_state(cfg.resume_from_ckpt)

    options = {
        "max_epoch": cfg["max_epoch"],
        "is_logging": is_logging,
        "log_every_n_steps": cfg["log_every_n_steps"],
        "ckpt_every_n_steps": cfg["ckpt_every_n_steps"],
        "ckpt_dir": cfg["ckpt_dir"],
        "fast_dev_run": cfg["fast_dev_run"],
    }
    
    print(f"=> Starting model training [epochs={cfg['max_epoch']}]")
    min_loss = None
    global_step = cfg.get("global_step", 0)
    for epoch in range(start_epoch, cfg["max_epoch"]):
        global_step = training_epoch(
            options=options,
            epoch=epoch,
            global_step=global_step,
            accelerator=accelerator,
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            discriminator_iter_start=discriminator_iter_start,
            default_metrics=default_metrics,
            rec_metrics=metrics,
            optimizer_ae=opt_ae,
            optimizer_disc=opt_disc,
        )

        accelerator.wait_for_everyone()

        loss = validation_epoch(
            options=options,
            epoch=epoch,
            accelerator=accelerator,
            dataloader=valid_dataloader,
            model=model,
            criterion=criterion,
            default_metrics=default_metrics,
            rec_metrics=metrics,
            global_step=global_step,
        )

        # save the best model
        if min_loss is None or loss < min_loss:
            try:
                accelerator.save_state(os.path.join(cfg.ckpt_dir, "best.pt"))
            except Exception as e:
                print(e)
            min_loss = loss

        # save checkpoint
        if (epoch + 1) % cfg.get("ckpt_every_n_epochs", 1) == 0:
            print(f"=> Saving checkpoint [epoch={epoch}]")
            try:
                accelerator.save_state(os.path.join(cfg.ckpt_dir, f"epoch-{epoch:04d}.pt"))
            except Exception as e:
                print(e)

    # save last model
    accelerator.save_state(os.path.join(cfg.ckpt_dir, "last.pt"))
    
    # save model to output directory
    accelerator.save(model, os.path.join(cfg.output, "model.pt"))

    print(f"=> Finished model training [epochs={cfg['max_epoch']}, metric={min_loss}]")
    accelerator.end_training()
    
if __name__ == "__main__":
    main()