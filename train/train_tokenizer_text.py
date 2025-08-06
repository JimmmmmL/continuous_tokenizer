# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import ruamel.yaml as yaml
import numpy as np
from tqdm import tqdm 
import torch.nn.functional as F
import os
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
from glob import glob
from copy import deepcopy
import torch.nn as nn
from timm.scheduler import create_scheduler_v2 as create_scheduler

from utils.logger_func import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.misc import str2bool, manage_checkpoints, load_model_state_dict
from utils.optim import param_groups_weight_decay
from utils.data_text import get_dataloaders
from modelling.bert_mae import TextMAE, TextMAEArgs
from transformers import GPT2TokenizerFast

import warnings
warnings.filterwarnings('ignore')
import difflib
import html
import re

# Split into words, spaces, and punctuation so we keep layout.
_TOK = re.compile(r'\w+|\s+|[^\w\s]')

def highlight_diff_fast_tokens(orig: str, recon: str) -> str:
    a = _TOK.findall(orig)
    b = _TOK.findall(recon)
    sm = difflib.SequenceMatcher(None, a, b)
    out = []
    add_open = '<span style="color:red">'
    add_close = '</span>'
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            out.append(html.escape(''.join(b[j1:j2])))
        elif tag in ('replace', 'insert'):
            seg = ''.join(b[j1:j2])
            if seg:
                out.append(add_open)
                out.append(html.escape(seg))
                out.append(add_close)
        # 'delete' => skip
    return ''.join(out)



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if args.exp_index is not None:
            experiment_index = int(args.exp_index)
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
        if args.config is not None:
            model_string_name = '.'.join(args.config.split('/')[-1].split('.')[:-1])
            if model_string_name.startswith('exp'):
                model_string_name = '-'.join(model_string_name.split('-')[1:])
        else:
            model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/exp{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        experiment_config = vars(args)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
        
        wandb_logger = wandb.init(project='maetok-text', name=f'exp{experiment_index:03d}-{model_string_name}')
    else:
        logger = create_logger(None)
        wandb_logger = None

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = TextMAE(
        config=TextMAEArgs(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            num_latent_tokens=args.num_latent_tokens,
            codebook_embed_dim=args.codebook_embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            width=args.width,
            token_drop_rate=args.token_drop_rate,
            token_drop_rate_max=args.token_drop_rate_max,
        )
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters() if p.requires_grad):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters() if p.requires_grad):,}")
    vq_model = vq_model.to(device)
    
    # scaling lr
    args.lr = args.lr * args.global_batch_size / 256
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups_weight_decay(vq_model, weight_decay=args.weight_decay), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # Setup data:
    
    train_loader, val_loader = get_dataloaders(config=args, logger=logger)

    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = args.epochs * num_update_steps_per_epoch

    # create lr scheduler
    if args.lr_scheduler == 'none':
        vqvae_lr_scheduler = None
    else:
        vqvae_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=optimizer,
            patience_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=args.lr_warmup_epochs,
            min_lr=args.lr * 0.1,
        )

    logger.info(f"num_update_steps_per_epoch {num_update_steps_per_epoch:,} max_train_steps ({max_train_steps})")

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        
        vq_model.load_state_dict(load_model_state_dict(checkpoint['model']))
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(train_loader.dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(train_loader.dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0       
        logger.info("compiling done.")
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    # Loss: Vanilla CE loss
    loss = nn.CrossEntropyLoss()

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', attn_implementation="sdpa")

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x in train_loader:
                
            input_ids = x['input_ids'].to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=ptdtype):
                recon_logits = vq_model(input_ids) # B, S, V
                # Cross Entropy loss
                # calculate loss, regardless of masked modelling                
                loss_gen = loss(recon_logits.view(-1, recon_logits.size(-1)), input_ids.view(-1))

                if train_steps % args.log_every == 0:
                    # calculate accuracy
                    acc = (recon_logits.argmax(dim=-1) == input_ids).float().mean().item()
                    logger.info(f"Step {train_steps}: Generator Loss: {loss_gen.item():.4f}, Accuracy: {acc:.4f}")
                # log to wandb
                if rank == 0  and wandb_logger is not None:
                    wandb_logger.log({"loss_gen": loss_gen.item(), "acc": acc}, step=train_steps)

            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)
            
            # # Log loss values:
            running_loss += loss_gen.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/total_steps:{max_train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()
            
                if rank == 0 and wandb_logger is not None:
                    log_dict = {"lr": optimizer.param_groups[0]["lr"], "train_loss": avg_loss}
                    wandb_logger.log(log_dict,
                        step=train_steps
                    )
                
            if train_steps % args.vis_every == 0:
                # Visualize text reconstructions:
                vq_model.eval()
                # Validation loop:
                val_losses, val_accs = [], []
                with torch.no_grad():
                    for idx, val_x in tqdm(enumerate(val_loader), total=len(val_loader), disable=(rank != 0)):
                        val_input_ids = val_x['input_ids'].to(device, non_blocking=True)
                        recon_logits = vq_model(val_input_ids)  # B, S, V
                    
                        # Calculate loss and accuracy
                        val_loss = loss(recon_logits.view(-1, recon_logits.size(-1)), val_input_ids.view(-1))
                        val_acc = (recon_logits.argmax(dim=-1) == val_input_ids).float().mean().item()
                        
                        val_losses.append(val_loss.item())
                        val_accs.append(val_acc)
                        

                        # Convert to text
                        if idx == 0:  # Only visualize the first batch
                            recon_ids = recon_logits.argmax(dim=-1)
                            logger.info(f"Visualizing reconstructions at step {train_steps}...")
                            original_texts = tokenizer.batch_decode(val_input_ids, skip_special_tokens=True)
                            recon_texts = tokenizer.batch_decode(recon_ids, skip_special_tokens=True)

                            table = wandb.Table(columns=["Original Text", "Reconstructed Text (Diff Highlighted)"])
                            logger.info("Constructing the reconstruction comparison table...")
                            for orig, recon in zip(original_texts, recon_texts):
                                recon_marked = highlight_diff_fast_tokens(orig, recon)
                                table.add_data(orig, wandb.Html(recon_marked))

                            logger.info("Logging reconstruction comparison table to wandb...")
                            if wandb_logger is not None:
                                wandb_logger.log({"Reconstruction Comparison": table}, step=train_steps)
                            logger.info("Reconstruction comparison logged to wandb.")
                            
                # Log validation loss and accuracy:
                avg_val_loss = np.mean(val_losses)
                avg_val_acc = np.mean(val_accs)
                logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}")
                if rank == 0 and wandb_logger is not None:
                    wandb_logger.log({"val_loss": avg_val_loss, "val_acc": avg_val_acc}, step=train_steps)
                
                vq_model.train()  # Switch back to training mode

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:

                if rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    # if not args.no_local_save:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    manage_checkpoints(checkpoint_dir)
                dist.barrier()

            if vqvae_lr_scheduler is not None:
                vqvae_lr_scheduler.step_update(train_steps)


    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/textmae-512-128.yaml', help="config file used to specify parameters")
    
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    parser.add_argument("--dataset", type=str, default="wikitext103", help="dataset to use for training")
    parser.add_argument("--cloud-save-path", type=str, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--codebook-embed-dim", type=int, default=32, help="codebook dimension for vector quantization")

    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_text")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none')
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--num-latent-tokens", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50257, help="vocab size for text tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=512, help="maximum sequence length for text tokenizer")
    parser.add_argument("--num_layers", type=int, default=12, help="number of layers/blocks in the encoder/decoder")
    parser.add_argument("--num-heads", type=int, default=12, help="number of attention heads in the encoder/decoder")
    parser.add_argument("--width", type=int, default=768, help="hidden dimension of the encoder/decoder")

    
    # mask modeling
    # make sure drop is 0.0 for not using mask modeling
    parser.add_argument("--token-drop-rate", type=float, default=0.4, help='encoder token drop')
    parser.add_argument("--token-drop-rate-max", type=float, default=0.6, help='maximum drop rate')

    # Evaluation
    parser.add_argument("--eval-batch-size", type=int, default=8, help="batch size for evaluation")

    # First parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    main(args)
