from torch.optim import AdamW
from tqdm import tqdm
import torch

class Trainer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def train_one_epoch(self, model, dataloader, optimizer, device, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (pixel_values, targets) in enumerate(progress_bar):
            # Move to device
            pixel_values = pixel_values.to(device)
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=targets)

            # Loss is computed internally by the model
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def train(self, model, train_dataloader, val_dataloader, optimizer, device, num_epochs):
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")

            # Train
            train_loss = self.train_one_epoch(
                model, train_dataloader, optimizer, device, epoch + 1)
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate(model, val_dataloader, device)
            self.val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"detr_checkpoint_epoch_{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")


    def validate(self, model, dataloader, device):
        """Validate the model."""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for pixel_values, targets in tqdm(dataloader, desc="Validation"):
                pixel_values = pixel_values.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

                outputs = model(pixel_values=pixel_values, labels=targets)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def reset_losses(self):
        self.train_losses = []
        self.val_losses = []
