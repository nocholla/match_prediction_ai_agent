import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import time
from src.data_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibilityDataset(Dataset):
    """Dataset for compatibility prediction."""
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class CompatibilityModel(nn.Module):
    """PyTorch neural network for compatibility prediction."""
    def __init__(self, input_dim, hidden_dims):
        super(CompatibilityModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def load_model_and_scaler(models_dir, model_file="pytorch_model.pth", scaler_file="scaler.pkl"):
    """Load PyTorch model and scaler if they exist."""
    model_path = os.path.join(models_dir, model_file)
    scaler_path = os.path.join(models_dir, scaler_file)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        logger.info("Loading existing PyTorch model and scaler")
        model = torch.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

def train_model(interaction_matrix, X_features):
    """
    Train PyTorch neural network for compatibility prediction.
    Returns: trained model, scaler
    """
    config = load_config()
    model_params = config['model_params']
    training_params = config['training_params']
    start_time = time.time()

    try:
        num_users, num_items = interaction_matrix.shape
        rows, cols = interaction_matrix.nonzero()
        values = interaction_matrix.data / 2.0  # Normalize to [0, 1]

        # Prepare features
        X_train = []
        for user_idx, item_idx in zip(rows, cols):
            features = X_features[item_idx].toarray().flatten()
            X_train.append(np.concatenate([[user_idx, item_idx], features]))
        X_train = np.array(X_train)
        y_train = values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Create dataset and dataloader
        dataset = CompatibilityDataset(X_scaled, y_train)
        dataloader = DataLoader(dataset, batch_size=training_params['batch_size'], shuffle=True)

        # Initialize model
        input_dim = X_scaled.shape[1]
        model = CompatibilityModel(input_dim, model_params['hidden_dims'])
        optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(training_params['epochs']):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{training_params['epochs']}, Loss: {total_loss/len(dataloader):.4f}")

        logger.info(f"Model Training Time: {time.time() - start_time:.2f} seconds")
        return model, scaler

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def predict_compatibility(model, scaler, user_id, filtered_profiles, X_features, user_to_idx, profile_to_idx):
    """
    Predict compatibility scores for filtered profiles.
    Returns: filtered_profiles with ml_score and final_score
    """
    try:
        item_indices = [profile_to_idx[pid] for pid in filtered_profiles['__id__'] if pid in profile_to_idx]
        if not item_indices:
            logger.error("No valid profiles for ML prediction")
            return filtered_profiles

        user_indices = np.full(len(item_indices), user_to_idx.get(user_id, 0))
        X_pred = [np.concatenate([[u, i], X_features[i].toarray().flatten()]) 
                  for u, i in zip(user_indices, item_indices)]
        X_pred = scaler.transform(X_pred)
        X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            scores = model(X_pred_tensor).squeeze().numpy()

        filtered_profiles['ml_score'] = 0.0
        profile_ids = filtered_profiles['__id__'].values
        for idx, score in zip(item_indices, scores):
            filtered_profiles.loc[filtered_profiles['__id__'] == profile_ids[idx], 'ml_score'] = score

        filtered_profiles['final_score'] = (filtered_profiles['ml_score'] * 0.7 + 
                                           filtered_profiles['country_match'].astype(int) * 0.1 + 
                                           filtered_profiles['language_match'].astype(int) * 0.1 + 
                                           filtered_profiles['goal_match'].astype(int) * 0.1)

        logger.info(f"Predicted {len(scores)} scores, mean: {np.mean(scores):.3f}, std: {np.std(scores):.3f}")
        return filtered_profiles

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise