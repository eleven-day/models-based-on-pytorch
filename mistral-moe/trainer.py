import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

class MoETrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader=None,
        optimizer=None,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        num_epochs=3,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        output_dir="./outputs",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        if optimizer is None:
            self.optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        
        # Move model to device
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train(self):
        """Main training loop"""
        total_train_loss = 0
        log_loss = 0
        start_time = time.time()
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print(f"\n === Epoch {epoch+1}/{self.num_epochs} ===")
            epoch_iterator = tqdm(self.train_dataloader, desc="Training")
            
            for step, batch in enumerate(epoch_iterator):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Scale loss for gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Track loss
                total_train_loss += loss.item()
                log_loss += loss.item()
                
                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Scheduler step
                    if self.scheduler is not None:
                        self.scheduler.step()
                        
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Increment global step
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        ms_per_step = elapsed * 1000 / self.logging_steps
                        avg_loss = log_loss / self.logging_steps
                        
                        print(f"Step {self.global_step}: loss = {avg_loss:.4f}, {ms_per_step:.2f} ms/step")
                        log_loss = 0
                        start_time = current_time
                    
                    # Evaluation
                    if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate()
                        print(f"Evaluation at step {self.global_step}: loss = {eval_loss:.4f}")
                        
                        # Save best model
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_model(os.path.join(self.output_dir, "best_model"))
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint-{self.global_step}"))
            
            # End of epoch
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} completed. Average train loss: {avg_train_loss:.4f}")
            
            # Save model after each epoch
            self.save_model(os.path.join(self.output_dir, f"epoch-{epoch+1}"))
            
        print("\nTraining completed!")
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model"))
        
        return {
            "global_step": self.global_step,
            "epochs": self.epoch + 1,
            "train_loss": total_train_loss / (len(self.train_dataloader) * self.num_epochs),
            "best_eval_loss": self.best_eval_loss
        }
    
    def evaluate(self):
        """Evaluate model on evaluation dataset"""
        if not self.eval_dataloader:
            raise ValueError("No evaluation dataloader provided")
        
        self.model.eval()
        total_eval_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                total_eval_loss += loss.item()
        
        avg_eval_loss = total_eval_loss / len(self.eval_dataloader)
        
        return avg_eval_loss
    
    def save_model(self, output_dir):
        """Save model and training state"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save training state
        training_state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_eval_loss": self.best_eval_loss
        }
        torch.save(training_state, os.path.join(output_dir, "training_state.bin"))
        
        # Save config
        config_dict = self.model.config.__dict__.copy()
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Model saved to {output_dir}")
        
    def load_model(self, model_path):
        """Load model and training state from a directory"""
        model_file = os.path.join(model_path, "pytorch_model.bin")
        state_file = os.path.join(model_path, "training_state.bin")
        
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            print(f"Model loaded from {model_file}")
            
        if os.path.exists(state_file):
            state = torch.load(state_file, map_location=self.device)
            self.epoch = state["epoch"]
            self.global_step = state["global_step"]
            self.best_eval_loss = state["best_eval_loss"]
            self.optimizer.load_state_dict(state["optimizer_state"])
            
            if self.scheduler and state["scheduler_state"]:
                self.scheduler.load_state_dict(state["scheduler_state"])
                
            print(f"Training state loaded from {state_file}")
