import os
import json
import random
import numpy as np
import torch
import pickle as pkl
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adafactor, AdamW
import pandas as pd

from utils.log import WandbLogger, CheckpointSaver, get_save_dir, load_model_checkpoint
from utils.config import args_parser
from utils.data import get_usage_dicts, load_dataset, eval_dataset_split
from dataloaders.poi_emb_dataset import POIEmbeddingDataset
from utils.fine_tuning import FineTuning
from modules.embed_tuner import EmbedTuner

# Device configuration
USE_GPU = True
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
print("Using device:", device)

# Verbosity setting
VERBOSE = True

def filter_by_available_embeddings(poi_ids, labels, embedding_dict):
    """
    Filter POI IDs and labels to keep only those for which we have embeddings.
    """
    available = set(embedding_dict.keys())
    keep_mask = np.array([int(pid) in available for pid in poi_ids])
    
    return poi_ids[keep_mask], labels[keep_mask]

def main():
    args = args_parser("config.json")
    
    model_names = ['e5', 'gtr-t5', 'gemini', 'nomic', 'mpnet', 'openai-large', 'openai-small']
    baseline_names = ['ctle', 'hier', 'deepmove', 'getnext', 'graphflashback', 'poi2vec', 
                      'skipgram', 'stan', 'tale', 'teaser', 'trajgpt', 'poi_embeds_openai-large']
    
    for name in baseline_names:
        all_results = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "auroc": [],
            "auprc": [],
            "mae": [],
            "cosine_sim": [],
        }
        run_name = f"Downstream Task:[{args.downstream_task}]" + \
                    f" City:[{args.city}]" + \
                    f" Model:[{name}]" + \
                    f" Mode:[{args.use_emb_type}]" + \
                    f" Time:[{datetime.now().strftime('%Y-%m-%d_%H-%M')}]"
        print("Run name:", run_name)

        for random_seed in args.random_seeds:
            print("Random seed:", random_seed)

            # Set random seeds for reproducibility
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

            # Setup WandB logger
            run_name_seed = f"{run_name} Seed:[{random_seed}]"
            wandb_logger = WandbLogger(args.project_name, args.use_wandb, run_name_seed, entity=args.entity)
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
        
            # Split by POI ids for evaluation. The whole sequencies are used for fine-tuning.
            if args.city == "LosAngeles":
                args.area_bbox = args.LA_area_bbox
                args.timezone = args.LA_timezone
            elif args.city == "Houston":
                args.area_bbox = args.Houston_area_bbox
                args.timezone = args.Houston_timezone
            
            # Check if external labels exist in args
            # If they do, then the evaluation will be on external data
            if args.use_external_labels:    
                df_data = pd.read_parquet(args.external_labels_path)
                print(f"Loaded external labels with shape: {df_data.shape}")
            else:
            # Else, load labels from Veraset x Safegraph dataset        
                df_data = load_dataset(
                    path=args.data_path, 
                    file_name=args.file_name,
                    loc_encoder_type=args.loc_encoder_type,
                    area_bbox=args.area_bbox,
                    area_timezone=args.timezone,
                )

            # Split the dataset into train, validation, and test sets
            df_train, df_other = eval_dataset_split(
                dataset=df_data,
                task=args.downstream_task,
                ratio=0.4, 
                seed=random_seed)
                    
            df_val, df_test = eval_dataset_split(
                dataset=df_other, 
                task=args.downstream_task,
                ratio=0.5, 
                seed=random_seed)
            
        
            # Prepare features and labels
            X_train = torch.tensor(np.stack(df_train['place_id'].to_numpy())).long()
            X_val = torch.tensor(np.stack(df_val['place_id'].to_numpy())).long()
            X_test = torch.tensor(np.stack(df_test['place_id'].to_numpy())).long()

            # Prepare labels based on the downstream task
            if args.downstream_task == "open_hours":
                y_train = np.stack(df_train['open_hours'].to_numpy())
                y_val = np.stack(df_val['open_hours'].to_numpy())
                y_test = np.stack(df_test['open_hours'].to_numpy())
            elif args.downstream_task == "busyness":
                y_train = np.stack(df_train['busyness'].to_numpy())
                y_val = np.stack(df_val['busyness'].to_numpy())
                y_test = np.stack(df_test['busyness'].to_numpy())
            elif args.downstream_task == "is_closed":
                y_train = df_train['is_closed'].to_numpy().astype(np.float32)
                y_val = df_val['is_closed'].to_numpy().astype(np.float32)
                y_test = df_test['is_closed'].to_numpy().astype(np.float32)
            else:
                raise ValueError(f"Unknown downstream task: {args.downstream_task}")
            print(f"Label shapes: train {y_train.shape}, val {y_val.shape}, test {y_test.shape}")
        
            # Load embedding from pickle file
            poi_embeddings = torch.load(args.emb_path + f"/baselines/{name}.pt")
            text_embeddings = torch.load(args.emb_path + f"text_embeds_openai-large.pt")
            
            # Filter out POIs without embeddings
            X_train, y_train = filter_by_available_embeddings(X_train, y_train, text_embeddings)
            X_val, y_val = filter_by_available_embeddings(X_val, y_val, text_embeddings)
            X_test, y_test = filter_by_available_embeddings(X_test, y_test, text_embeddings)

            # Convert to tensors
            X_train = torch.tensor(X_train)
            X_val = torch.tensor(X_val)
            X_test = torch.tensor(X_test)

            y_train = torch.tensor(y_train)
            y_val = torch.tensor(y_val)
            y_test = torch.tensor(y_test)

            # Adjust label dimensions for binary classification
            if args.downstream_task == "is_closed":
                y_train = y_train.unsqueeze(1)
                y_val = y_val.unsqueeze(1)
                y_test = y_test.unsqueeze(1)

            # Create data loaders
            train_loader = DataLoader(
                POIEmbeddingDataset(X_train, y_train, poi_embeddings, text_embeddings),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True
            )
            
            val_loader = DataLoader(
                POIEmbeddingDataset(X_val, y_val, poi_embeddings, text_embeddings),
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False
            )
            
            test_loader = DataLoader(
                POIEmbeddingDataset(X_test, y_test, poi_embeddings, text_embeddings),
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False
            )
            
            # Initialize the model
            ft_model = EmbedTuner(
                poi_embed_dim=args.dim_embed,
                text_embed_dim=args.dim_text_embed,
                num_classes=args.d_out,
                hidden_dim=256,
                dropout_rate=0.5
            ).to(device)

            optimizer = AdamW(ft_model.parameters(), lr=args.fine_tune_lr, weight_decay=args.weight_decay)
            
            if args.downstream_task == "open_hours" or args.downstream_task == "busyness":
                # We treat open hours as a multi-label classification task
                criterion = nn.BCEWithLogitsLoss()
            
            elif args.downstream_task == "is_closed":
                # We treat is_closed as a binary classification task
                n_pos = df_data[df_data['safegraph.closed_on'].notna()]['place_id'].nunique()
                print(f"Number of positive samples: {n_pos}")
                print(f"Number of negative samples: {df_data['place_id'].nunique() - n_pos}")
                total = df_data['place_id'].nunique()
                pos_weight = total / n_pos
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=dtype).to(device))
            else:
                raise ValueError(f"Unknown downstream task: {args.downstream_task}")

            # Fine tuning and evaluation
            torch.autograd.set_detect_anomaly(True)
            tuner = FineTuning(
                args, ft_model, optimizer, criterion, device, task=args.downstream_task,
                wandb_logger=wandb_logger, checkpoint_saver=checkpoint_saver, mode=args.use_emb_type
            )
            tuner.run(
                train_loader, val_loader, epochs=args.fine_tune_epochs, verbose=VERBOSE, 
                patience=10 
            )
            
            # Load best model for evaluation
            print("Loading best model for evaluation...")
            best_path = os.path.join(save_dir, "best.pth.tar")
            best_model = load_model_checkpoint(best_path, ft_model)
            
            metrics = tuner.eval_epoch(test_loader, best_model)
            print("Test metrics:", metrics)

            # Add metric results for this seed to dictionary
            for metric in metrics.keys():
                if metric in all_results.keys():
                    all_results[metric].append(metrics[metric])
        
        # Aggregate results across all seeds and append to a file
        results_file = os.path.join(args.save_dir, f"results_{args.downstream_task}_{args.city}_{args.use_emb_type}_baselines.txt")
        with open(results_file, "a") as f:
            f.write(f"Results for run: {run_name}\n")
            for metric in all_results.keys():
                # check that the metric has results
                if all_results[metric] != []:
                    f.write(f"{metric}: {np.mean(all_results[metric])} +- {np.std(all_results[metric])}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
