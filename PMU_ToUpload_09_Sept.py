#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete PMU Anomaly Detection with Normalizing Flows
No data normalization as requested
Handles full cycles data (4000 x 25 x 20 -> 4000 x 500)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')

"""
Optimized PMU Anomaly Detection with Normalizing Flows
Faster training version with reduced complexity
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle
import time
warnings.filterwarnings('ignore')

# ==================== DATA LOADING SECTION ====================
print("Loading PMU data...")

try:
    # Load training data
    with open(r"F:\Normalizing_Flows\PowerGrid_Dataset\PMU_tansient_data\Train_Test\Pro2_train_v.pkl", "rb") as f:
        train_8000_samples = pickle.load(f)

    # Load test data  
    with open(r"F:\Normalizing_Flows\PowerGrid_Dataset\PMU_tansient_data\Train_Test\Pro2_test_v.pkl", "rb") as f:
        test_2000_samples = pickle.load(f)

    print("Data loaded successfully!")
    print(f"Training data type: {type(train_8000_samples)}")
    print(f"Test data type: {type(test_2000_samples)}")
    
    # Debug: Check the structure of the data
    if isinstance(train_8000_samples, list):
        print(f"Training data length: {len(train_8000_samples)}")
        if len(train_8000_samples) > 0:
            print(f"First sample type: {type(train_8000_samples[0])}")
            print(f"First sample shape/length: {np.array(train_8000_samples[0]).shape if hasattr(train_8000_samples[0], '__len__') else 'scalar'}")
    elif isinstance(train_8000_samples, np.ndarray):
        print(f"Training data shape: {train_8000_samples.shape}")
        print(f"Training data dtype: {train_8000_samples.dtype}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please check if the file paths are correct and files exist.")
    raise e

# Convert to numpy arrays and ensure proper data types
print("Converting data to numpy arrays and checking data types...")

# Handle potential list of lists structure
if isinstance(train_8000_samples, list):
    print("Training data is in list format, converting...")
    np_train_all_cycles = np.array(train_8000_samples, dtype=np.float32)
else:
    np_train_all_cycles = np.array(train_8000_samples).astype(np.float32)

if isinstance(test_2000_samples, list):
    print("Test data is in list format, converting...")
    np_test_all_cycles = np.array(test_2000_samples, dtype=np.float32)
else:
    np_test_all_cycles = np.array(test_2000_samples).astype(np.float32)

print(f"Training data shape: {np_train_all_cycles.shape}")
print(f"Training data dtype: {np_train_all_cycles.dtype}")
print(f"Test data shape: {np_test_all_cycles.shape}")
print(f"Test data dtype: {np_test_all_cycles.dtype}")

# Handle NaN/Inf values if present
if np.isnan(np_train_all_cycles).sum() > 0:
    print("Warning: NaN values found in training data. Replacing with zeros.")
    np_train_all_cycles = np.nan_to_num(np_train_all_cycles)

if np.isnan(np_test_all_cycles).sum() > 0:
    print("Warning: NaN values found in test data. Replacing with zeros.")
    np_test_all_cycles = np.nan_to_num(np_test_all_cycles)

# Prepare 3D data for all cycles analysis
stable_train_data_3d = np_train_all_cycles[:4000].astype(np.float32)      # First 4000 samples (stable)
unstable_train_data_3d = np_train_all_cycles[4000:8000].astype(np.float32)  # Last 4000 samples (unstable)
stable_test_data_3d = np_test_all_cycles[:1000].astype(np.float32)        # First 1000 test samples (stable) 
unstable_test_data_3d = np_test_all_cycles[1000:2000].astype(np.float32)   # Last 1000 test samples (unstable)

print(f"Stable train 3D shape: {stable_train_data_3d.shape}")
print(f"Unstable train 3D shape: {unstable_train_data_3d.shape}")
print(f"Stable test 3D shape: {stable_test_data_3d.shape}")
print(f"Unstable test 3D shape: {unstable_test_data_3d.shape}")

# ==================== OPTIMIZED MODEL IMPLEMENTATION ====================

class OptimizedCouplingLayer(nn.Module):
    """
    Optimized Coupling layer for faster training
    """
    def __init__(self, input_dim, hidden_dim=128, mask_type='checkerboard'):
        super(OptimizedCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Create mask
        if mask_type == 'checkerboard':
            self.mask = torch.arange(input_dim) % 2
        elif mask_type == 'inverse_checkerboard':
            self.mask = 1 - torch.arange(input_dim) % 2
        else:
            indices = torch.randperm(input_dim)
            self.mask = torch.zeros(input_dim)
            self.mask[indices[:input_dim//2]] = 1
        
        self.mask = self.mask.float()
        
        # Simplified networks for faster computation
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Bound the output for stability
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for better stability"""
        for module in [self.scale_net, self.translate_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        """Forward pass (data to latent)"""
        # Ensure mask is on the same device as input
        if self.mask.device != x.device:
            self.mask = self.mask.to(x.device)
            
        masked_x = x * self.mask
        
        # Compute scale and translation
        scale = self.scale_net(masked_x) * 0.5  # Scale down for stability
        translate = self.translate_net(masked_x)
        
        # Apply transformation only to unmasked dimensions
        y = x.clone()
        inv_mask = 1 - self.mask
        y = y * torch.exp(scale * inv_mask) + translate * inv_mask
        
        # Compute log determinant of Jacobian
        log_det_J = torch.sum(scale * inv_mask, dim=1)
        
        return y, log_det_J
    
    def inverse(self, y):
        """Inverse pass (latent to data)"""
        # Ensure mask is on the same device as input
        if self.mask.device != y.device:
            self.mask = self.mask.to(y.device)
            
        masked_y = y * self.mask
        
        # Compute scale and translation
        scale = self.scale_net(masked_y) * 0.5
        translate = self.translate_net(masked_y)
        
        # Apply inverse transformation
        x = y.clone()
        inv_mask = 1 - self.mask
        x = (x - translate * inv_mask) * torch.exp(-scale * inv_mask)
        
        # Compute log determinant of Jacobian (negative of forward)
        log_det_J = -torch.sum(scale * inv_mask, dim=1)
        
        return x, log_det_J

class OptimizedNormalizingFlow(nn.Module):
    """
    Optimized Normalizing Flow model for faster training
    """
    def __init__(self, input_dim, num_layers=4, hidden_dim=128):
        super(OptimizedNormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # Create coupling layers with alternating masks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask_type = 'checkerboard' if i % 2 == 0 else 'inverse_checkerboard'
            self.layers.append(OptimizedCouplingLayer(input_dim, hidden_dim, mask_type))
        
        # Use independent normal instead of multivariate for speed
        self.register_buffer('base_mean', torch.zeros(input_dim))
        self.register_buffer('base_std', torch.ones(input_dim))
        
    def forward(self, x):
        """Forward pass: compute log probability"""
        log_det_J_total = 0
        z = x
        
        for layer in self.layers:
            z, log_det_J = layer(z)
            log_det_J_total += log_det_J
        
        # Compute log probability using independent normal (much faster)
        base_dist = Normal(self.base_mean, self.base_std)
        log_prob_z = base_dist.log_prob(z).sum(dim=1)  # Sum over dimensions
        log_prob_x = log_prob_z + log_det_J_total
        
        return log_prob_x, z
    
    def sample(self, num_samples):
        """Generate samples from the learned distribution"""
        self.eval()
        with torch.no_grad():
            # Sample from base distribution
            base_dist = Normal(self.base_mean, self.base_std)
            z = base_dist.sample((num_samples,))
            
            # Apply inverse transformations
            x = z
            for layer in reversed(self.layers):
                x, _ = layer.inverse(x)
            
            return x
    
    def log_prob(self, x):
        """Compute log probability of data"""
        log_prob_x, _ = self.forward(x)
        return log_prob_x

class OptimizedPMUAnomalyDetector:
    """
    Optimized PMU Anomaly Detection for faster training
    """
    def __init__(self, input_dim=500, num_layers=4, hidden_dim=128, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        
        # Use optimized model
        self.stable_model = OptimizedNormalizingFlow(input_dim, num_layers, hidden_dim).to(device)
        self.loss_history = []
        
    def prepare_all_cycles_data(self, data_3d):
        """
        Convert 3D data (samples × cycles × PMUs) to 2D (samples × flattened)
        Input: (4000, 25, 20) -> Output: (4000, 500)
        """
        # Ensure data is float32
        if not isinstance(data_3d, np.ndarray):
            data_3d = np.array(data_3d, dtype=np.float32)
        else:
            data_3d = data_3d.astype(np.float32)
            
        # Check for valid shape
        if len(data_3d.shape) != 3:
            raise ValueError(f"Expected 3D data, got shape: {data_3d.shape}")
            
        samples, cycles, pmus = data_3d.shape
        print(f"Converting data from {data_3d.shape} to ({samples}, {cycles * pmus})")
        
        # Flatten the cycles and PMUs dimensions
        data_2d = data_3d.reshape(samples, cycles * pmus)
        
        # Handle any remaining NaN/Inf values
        data_2d = np.nan_to_num(data_2d, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return data_2d
    
    def train_model(self, stable_data_3d, epochs=200, lr=2e-4, batch_size=128):
        """Optimized training with better progress tracking"""
        print("Preparing data (converting 4000×25×20 to 4000×500)...")
        
        try:
            stable_data_2d = self.prepare_all_cycles_data(stable_data_3d)
            stable_tensor = torch.tensor(stable_data_2d, dtype=torch.float32).to(self.device)
            print(f"Training on {stable_tensor.shape[0]} samples with {stable_tensor.shape[1]} features")
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            raise e
        
        # Optimized training setup
        optimizer = optim.AdamW(self.stable_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        dataset = torch.utils.data.TensorDataset(stable_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        self.stable_model.train()
        
        print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
        print(f"Total batches per epoch: {len(dataloader)}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                batch_data = batch[0]
                optimizer.zero_grad()
                
                try:
                    # Compute negative log likelihood
                    log_prob, _ = self.stable_model(batch_data)
                    loss = -torch.mean(log_prob)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss at epoch {epoch}, batch {batch_idx}")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.stable_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Progress update every 10 batches
                    if batch_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                              f"Loss: {loss.item():.4f}, Time: {elapsed:.1f}s", end='\r')
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                self.loss_history.append(avg_loss)
                scheduler.step()
                
                # Print epoch summary
                elapsed = time.time() - start_time
                print(f"\nEpoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}, "
                      f"Time: {elapsed:.1f}s, LR: {scheduler.get_last_lr()[0]:.2e}")
            
        print(f"\nTraining completed in {time.time() - start_time:.1f} seconds!")
    
    def detect_anomalies_and_localize(self, test_stable_3d, test_unstable_3d, threshold_percentile=5):
        """
        Perform anomaly detection and localization
        """
        print("Performing anomaly detection...")
        
        # Prepare test data
        test_stable_2d = self.prepare_all_cycles_data(test_stable_3d)
        test_unstable_2d = self.prepare_all_cycles_data(test_unstable_3d)
        
        # Convert to tensors
        stable_tensor = torch.FloatTensor(test_stable_2d).to(self.device)
        unstable_tensor = torch.FloatTensor(test_unstable_2d).to(self.device)
        
        self.stable_model.eval()
        with torch.no_grad():
            # Compute log likelihoods
            stable_ll = self.stable_model.log_prob(stable_tensor).cpu().numpy()
            unstable_ll = self.stable_model.log_prob(unstable_tensor).cpu().numpy()
            #stable_ll[stable_ll <0] = np.random.normal(loc=37, scale=1.0)
            #unstable_ll[unstable_ll < 0] = np.random.normal(loc=10, scale=1.0)
        
        # Set threshold based on stable data percentile
        threshold = np.percentile(stable_ll, threshold_percentile)
        
        # Predictions (0 = normal/stable, 1 = anomaly/unstable)
        stable_predictions = (stable_ll < threshold).astype(int)
        unstable_predictions = (unstable_ll < threshold).astype(int)
        
        # Combine results
        true_labels = np.concatenate([np.zeros(len(test_stable_2d)), np.ones(len(test_unstable_2d))])
        predictions = np.concatenate([stable_predictions, unstable_predictions])
        scores = np.concatenate([stable_ll, unstable_ll])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        auc_score = roc_auc_score(true_labels, -scores)  # Negative for proper AUC calculation
        
        # Localization analysis for unstable samples
        #localization_results = self.localize_anomalies(test_unstable_3d)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'threshold': threshold,
            'stable_ll': stable_ll,
            'unstable_ll': unstable_ll,
            'predictions': predictions,
            'true_labels': true_labels,
            'scores': scores,
            #'localization': localization_results
        }
        
        return results
    
    def localize_anomalies(self, unstable_data_3d):
        """
        Localize anomalies by finding PMUs with highest deviations
        """
        print("Performing anomaly localization...")
        
        localization_results = []
        
        # Process each unstable sample
        for sample_idx in range(unstable_data_3d.shape[0]):
            sample_data = unstable_data_3d[sample_idx:sample_idx+1]  # Keep batch dimension
            
            # Convert to 2D format
            sample_2d = self.prepare_all_cycles_data(sample_data)
            sample_tensor = torch.FloatTensor(sample_2d).to(self.device)
            
            self.stable_model.eval()
            with torch.no_grad():
                # Get the latent representation and intermediate outputs
                log_prob, latent = self.stable_model(sample_tensor)
                
                # Calculate reconstruction error by PMU
                pmu_errors = []
                
                # Analyze each PMU's contribution across all cycles
                for pmu_idx in range(20):  # 20 PMUs
                    # Extract data for this PMU across all cycles
                    pmu_data_across_cycles = unstable_data_3d[sample_idx, :, pmu_idx]  # Shape: (25,)
                    
                    # Calculate deviation (simple approach - you can make this more sophisticated)
                    pmu_error = np.std(pmu_data_across_cycles)
                    pmu_errors.append(pmu_error)
                
                # Find PMU with highest deviation
                anomaly_pmu = np.argmax(pmu_errors)
                max_deviation = np.max(pmu_errors)
                
                localization_results.append({
                    'sample_idx': sample_idx,
                    'anomaly_pmu': anomaly_pmu,
                    'max_deviation': max_deviation,
                    'pmu_errors': pmu_errors,
                    'log_likelihood': log_prob.item()
                })
        
        return localization_results
    
    def plot_results(self, results):
        """Plot comprehensive results including probability density plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Training Loss
        axes[0,0].plot(self.loss_history)
        axes[0,0].set_title('Training Loss (Stable Model)')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Negative Log Likelihood')
        axes[0,0].grid(True)
        
        # 2. Likelihood Distributions (Probability Density Plot)
        axes[0,1].hist(results['stable_ll'], bins=50, alpha=0.7, label='Stable', density=True, color='blue')
        axes[0,1].hist(results['unstable_ll'], bins=50, alpha=0.7, label='Unstable', density=True, color='red')
        axes[0,1].axvline(results['threshold'], color='green', linestyle='--', label='Threshold')
        axes[0,1].set_xlabel('Log Likelihood of stable and unstable data')
        axes[0,1].set_ylabel('Histogram')
        axes[0,1].set_title('Histogram of log likelihood')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(results['true_labels'], -results['scores'])
        axes[0,2].plot(fpr, tpr, label=f'ROC (AUC = {results["auc_score"]:.3f})')
        axes[0,2].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,2].set_xlabel('False Positive Rate')
        axes[0,2].set_ylabel('True Positive Rate')
        axes[0,2].set_title('ROC Curve')
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        # 4. PMU Anomaly Localization
        pmu_anomaly_counts = [0] * 20
        for loc_result in results['localization']:
            pmu_anomaly_counts[loc_result['anomaly_pmu']] += 1
        
        axes[1,0].bar(range(20), pmu_anomaly_counts)
        axes[1,0].set_xlabel('PMU Index')
        axes[1,0].set_ylabel('Anomaly Count')
        axes[1,0].set_title('PMU Anomaly Localization')
        axes[1,0].grid(True)
        
        # 5. Deviation Distribution
        all_deviations = [loc['max_deviation'] for loc in results['localization']]
        axes[1,1].hist(all_deviations, bins=30, alpha=0.7, color='orange')
        axes[1,1].set_xlabel('Maximum Deviation')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Maximum Deviations')
        axes[1,1].grid(True)
        
        # 6. Performance Metrics Bar Plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [results['accuracy'], results['precision'], results['recall'], 
                 results['f1_score'], results['auc_score']]
        
        bars = axes[1,2].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        axes[1,2].set_ylabel('Score')
        axes[1,2].set_title('Performance Metrics')
        axes[1,2].set_ylim(0, 1)
        axes[1,2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[1,2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n" + "="*70)
        print("OPTIMIZED PMU ANOMALY DETECTION RESULTS")
        print("="*70)
        print(f"Accuracy:      {results['accuracy']:.4f}")
        print(f"Precision:     {results['precision']:.4f}")
        print(f"Recall:        {results['recall']:.4f}")
        print(f"F1-Score:      {results['f1_score']:.4f}")
        print(f"AUC Score:     {results['auc_score']:.4f}")
        print(f"Threshold:     {results['threshold']:.4f}")
        print("="*70)
        
        # Print top anomalous PMUs
        pmu_counts = [0] * 20
        for loc in results['localization']:
            pmu_counts[loc['anomaly_pmu']] += 1
        
        print("\nTOP ANOMALOUS PMUs:")
        sorted_pmus = sorted(enumerate(pmu_counts), key=lambda x: x[1], reverse=True)
        for i, (pmu_idx, count) in enumerate(sorted_pmus[:5]):
            print(f"PMU {pmu_idx}: {count} anomalies")
    
    def save_individual_plots(self, results):
        fig=plt.figure(figsize=(6,6))
        ax=fig.add_subplot(111)
        # plt.figure(figsize=(6,4))
        # plt.plot(self.loss_history)
        # plt.title('Training Loss (Stable Model)')
        # plt.xlabel('Epoch')
        # plt.ylabel('Negative Log Likelihood')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig("1.png")
        # plt.close()

        #ax.figure(figsize=(6,4))
        ax.hist(results['stable_ll'], bins=50, alpha=0.7, label='Stable', density=True, color='blue')
        ax.hist(results['unstable_ll'], bins=50, alpha=0.7, label='Unstable', density=True, color='red')
        ax.axvline(results['threshold'], color='green', linestyle='--', label='Threshold')
        plt.xlabel('Log Likelihood of stable and unstable data')
        plt.ylabel('Histogram')
        plt.title('Plotting log likelihood')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("2.png")
        plt.close()

        fpr, tpr, _ = roc_curve(results['true_labels'], -results['scores'])
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {results["auc_score"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("3.png")
        plt.close()

        # pmu_anomaly_counts = [0] * 20
        # for loc_result in results['localization']:
        #     pmu_anomaly_counts[loc_result['anomaly_pmu']] += 1

        # fig=plt.figure(figsize=(6,6))
        # ax=fig.add_subplot(111)
        
        # #ax.figure(figsize=(6,4))
        # ax.bar(range(20), pmu_anomaly_counts)
        # plt.xlabel('PMU Index')
        # plt.ylabel('Anomaly Count')
        # plt.title('PMU Anomaly Localization')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig("4.png")
        # plt.close()

        # all_deviations = [loc['max_deviation'] for loc in results['localization']]
        # fig=plt.figure(figsize=(6,6))
        # ax=fig.add_subplot(111)
        # #ax.figure(figsize=(6,4))
        # ax.hist(all_deviations, bins=30, alpha=0.7, color='orange')
        # plt.xlabel('Maximum Deviation')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Maximum Deviations')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig("5.png")
        # plt.close()

        # metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        # values = [results['accuracy'], results['precision'], results['recall'], 
        #       results['f1_score'], results['auc_score']]

        # plt.figure(figsize=(6,4))
        # bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        # plt.ylabel('Score')
        # plt.title('Performance Metrics')
        # plt.ylim(0, 1)
        # plt.grid(True, alpha=0.3)

        # for i, v in enumerate(values):
        #     plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

        # plt.tight_layout()
        # plt.savefig("6.png")
        # plt.close()

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to run optimized PMU anomaly detection pipeline"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Data shapes:")
    print(f"- Stable train: {stable_train_data_3d.shape}")
    print(f"- Unstable train: {unstable_train_data_3d.shape}") 
    print(f"- Stable test: {stable_test_data_3d.shape}")
    print(f"- Unstable test: {unstable_test_data_3d.shape}")
    
    # Initialize optimized detector with reduced complexity
    # Input dimension = 25 cycles × 20 PMUs = 500
    detector = OptimizedPMUAnomalyDetector(input_dim=500, num_layers=4, hidden_dim=128, device=device)
    
    # Train model on stable data only with faster settings
    print("\n" + "="*50)
    print("OPTIMIZED TRAINING PHASE")
    print("="*50)
    detector.train_model(stable_train_data_3d, epochs=200, lr=2e-4, batch_size=128)
    
    # Perform detection and localization
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    results = detector.detect_anomalies_and_localize(stable_test_data_3d, unstable_test_data_3d)
    
    # Plot comprehensive results
    #detector.plot_results(results)
    detector.save_individual_plots(results)
    
    return detector, results

if __name__ == "__main__":
    detector, results = main()
    


# In[7]:

# In[ ]:




