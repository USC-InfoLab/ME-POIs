import os
import json
import random
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from utils.criteria import Criterion

from torch.utils.data import DataLoader
from torch.optim import Adafactor

from utils.log import WandbLogger, CheckpointSaver, get_save_dir
from utils.config import args_parser
from utils.data import load_dataset, pretrain_dataset_split, get_usage_dicts
from dataloaders.visitseq_dataset import VisitSequencesDataset, collate_visit_sequences
from modules.visit_encoder import VisitEncoder
from utils.pretraining_cl import CL_Pretrainer
from utils.pretraining_mlm import MLM_Pretrainer


# Device configuration
USE_GPU = True
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
print("Using device:", device)

# Verbosity setting
VERBOSE = True

def main():
    args = args_parser("config.json")

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seeds[0])
    torch.cuda.manual_seed(args.random_seeds[0])
    torch.cuda.manual_seed_all(args.random_seeds[0])
    np.random.seed(args.random_seeds[0])
    random.seed(args.random_seeds[0])

    # Setup WandB logger
    run_name = f"{args.test_setting}_{args.city}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    print("Run name:", run_name)
    
    wandb_logger = WandbLogger(args.project_name, args.use_wandb, run_name, entity=args.entity)
    wandb_logger.log_hyperparams(vars(args))

    # Create save directory
    save_dir = get_save_dir(args.save_dir, training=True)
    os.makedirs(save_dir, exist_ok=True)

    # Save args to a JSON file
    args_file = os.path.join(save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Initialize checkpoint saver
    checkpoint_saver = CheckpointSaver(
        save_dir, metric_name="eloss", maximize_metric=False
    )
    
    # Load dataset. This function should return a preprocessed dataframe.
    if args.city == "LosAngeles":
        args.area_bbox = args.LA_area_bbox
        args.timezone = args.LA_timezone
    elif args.city == "Houston":
        args.area_bbox = args.Houston_area_bbox
        args.timezone = args.Houston_timezone
    
    dataset = load_dataset(
        path=args.data_path, 
        file_name=args.file_name,
        loc_encoder_type=args.loc_encoder_type,
        area_bbox=args.area_bbox,
        area_timezone=args.timezone
    )
    
    text_embeds = torch.load(args.emb_path + f"text_embeds_{args.text_model_name}.pt")
    print(f"Text embeddings keys sample: {list(text_embeds.keys())[:5]}")
    print(f"Loaded text embeddings from {args.emb_path + f'text_embeds_{args.text_model_name}.pt'}")
    
    # Compute or load precomputed anchor-based usage weights
    anchor_map, sparse_multiscale_map, sparse_text_sim_map = get_usage_dicts(
        dir_path=args.data_path,
        file_path=args.anchor_path,
        dataset=dataset,
        sigmas=args.gaussian_sigmas,
        text_embeds=text_embeds, # pass text_embeds to use text similarity distributions
        city=args.city,
        area_bbox=args.area_bbox,
        recompute= not args.anchor_precomputed_weights,
        distr_col='weekly'
    )
    
    if args.hyperparameter_tuning:
        # For hyperparameter tuning, we take out a small validation set
        train_dataset, val_dataset = pretrain_dataset_split(
            dataset, val_ratio=0.15, seed=args.random_seeds[0])
    
        print(f"Train dataset size: {(train_dataset.shape)}")
        print(f"Val dataset size: {(val_dataset.shape)}")
        
        train_loader = DataLoader(
            VisitSequencesDataset(train_dataset, args.window_size, args.dim_text_embed, args.emb_path + f"text_embeds_{args.text_model_name}.pt"),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_visit_sequences,
            drop_last=True
        )
        
        val_loader = DataLoader(
            VisitSequencesDataset(val_dataset, args.window_size, args.dim_text_embed, args.emb_path + f"text_embeds_{args.text_model_name}.pt"),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_visit_sequences,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            VisitSequencesDataset(dataset, args.window_size, args.dim_text_embed, args.emb_path + f"text_embeds_{args.text_model_name}.pt"),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_visit_sequences,
            drop_last=True
        )
        
        val_loader = None  # No validation set for full dataset pretraining
    
    model = VisitEncoder(
        dim_embed = args.dim_embed,
        num_heads = args.num_heads,
        dim_feedforward = args.dim_feedforward,
        dropout = args.dropout,
        num_layers = args.num_layers,
        num_pois = args.num_pois,
        num_categories = args.num_categories,
        loc_encoder_type = args.loc_encoder_type,
        strategy = args.pretraining_strategy,
        init_embeds = None, # do not preinitialize with pretrained POI embeddings
        args = args
    ).to(device)
    
    optimizer = Adafactor(list(model.parameters()))
    criterion = Criterion(num_categories=args.num_categories)

    # Pretraining
    torch.autograd.set_detect_anomaly(True)
    
    if args.pretraining_strategy == "CL":
        # Contrastive Learning pretraining
        pretrainer = CL_Pretrainer(
            model,
            optimizer,
            criterion,
            anchor_distr=anchor_map,
            sparse_distr=sparse_multiscale_map,
            sparse_text_sim_distr=sparse_text_sim_map,
            sigmas=args.gaussian_sigmas,
            device=device,
            log_vars=None,
            use_text_emb=False,
            wandb_logger=wandb_logger,
            checkpoint_saver=checkpoint_saver,
        )
        print(f"Number of trainable parameters: {pretrainer.count_params()}")
        
    elif args.pretraining_strategy == "MLM":
        # Masked Language Modeling pretraining
        pretrainer = MLM_Pretrainer(
            model, optimizer, criterion, device,
            wandb_logger=wandb_logger,
            checkpoint_saver=checkpoint_saver
        )
    
    pretrainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        verbose=VERBOSE,
        patience=10
    )
    
    if args.save_pretrained_model:
        pretrainer.save_embeddings(args.emb_path + f"poi_embeds.pt", save_text_emb=True, loader=train_loader)

if __name__ == "__main__":
    main()