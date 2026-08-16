import torch
from tqdm import tqdm
from utils.modeling import generate_mlm_mask

class MLM_Pretrainer:
    def __init__(self, model, optimizer, criterion, device, wandb_logger=None, checkpoint_saver=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.wandb_logger = wandb_logger
        self.checkpoint_saver = checkpoint_saver

        # Early stopping trackers
        self.best_loss = float('inf')
        self.patience_counter = 0

    def move_to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def log(self, metrics, epoch):
        if self.wandb_logger:
            for key, value in metrics.items():
                if value is not None:
                    self.wandb_logger.log(key, value, epoch)

    def print(self, metrics, epoch):
        msg = f"Epoch {epoch}: " + " ".join([f"{k}={v:.4f}" for k, v in metrics.items() if v is not None])
        print(msg)

    def early_stop_check(self, val_loss, epoch):
        """Update best loss and patience counter."""
        if val_loss is None:
            return False  # Nothing to check
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            # Save the model checkpoint
            self.checkpoint_saver.save(epoch, self.model, self.optimizer, val_loss)
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience_limit

    def run(self, train_loader, val_loader=None, epochs=100, verbose=True, patience=10):
        """Main pretraining loop."""
        self.patience_limit = patience
        if self.wandb_logger:
            self.wandb_logger.watch_model(self.model)

        for epoch in range(epochs):
            tloss = self.run_epoch(train_loader, train=True)
            if val_loader:
                print("Validation...")
                eloss = self.run_epoch(val_loader, train=False)

            # Log metrics
            metrics = {
                "Pretrain/Loss": tloss['total'],
                "Pretrain/LR": self.optimizer.param_groups[0]["lr"],
            }
            
            if val_loader:
                metrics.update({
                    "Eval/Loss": eloss['total'],
                })
            
            self.log(metrics, epoch)
            
            # Print metrics
            if verbose:
                self.print(metrics, epoch)

            # Early stopping check
            if self.early_stop_check(tloss['total'], epoch):
                print(f"Early stopping triggered at epoch {epoch}")
                break

    def run_epoch(self, visit_seq_loader, train=True):
        total_loss = 0
        self.model.train() if train else self.model.eval()

        context = torch.no_grad() if not train else torch.enable_grad()
        with context:
            for batch in tqdm(visit_seq_loader, desc="Training" if train else "Evaluating"):
                batch = self.move_to_device(batch)

                if train:
                    self.optimizer.zero_grad()

                # Generate MLM mask
                mlm_mask = generate_mlm_mask(shape=batch['place_id'].shape, 
                                            device=self.device,
                                            mask_prob=0.25)
                
                # Encode sequence using MLM
                seq_enc = self.model(batch, mlm_mask=mlm_mask)

                # Predict poi ids
                preds  = self.model.mlm_predict(seq_enc, batch, only_poi_id=False)
                
                # Compute contrastive learning loss
                true_vals = (batch['place_id'], batch['location'], batch['travel_time'], batch['duration'])
                loss = self.criterion.mlm_loss(preds, true_vals, mlm_mask)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        return {
            "total": total_loss / len(visit_seq_loader),
        }

    def save_embeddings(self, path):
        """
        Store the learned POI embeddings to a file.
        """
        self.model.eval()
        embeddings_dict = self.model.get_poi_embeddings()
        # Save it as pt
        torch.save(embeddings_dict, path)
        print(f"Saved POI embeddings to {path}")