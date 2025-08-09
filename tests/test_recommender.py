import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import torch
from scipy.sparse import csr_matrix
from src.recommender import train_model, predict_compatibility, CompatibilityModel, CompatibilityDataset
from src.data_loader import load_config

@pytest.fixture
def sample_data():
    config = load_config()
    interaction_matrix = csr_matrix((np.array([1, 2]), ([0, 1], [0, 1])), shape=(2, 2))
    X_features = csr_matrix(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
    user_to_idx = {'user1': 0, 'user2': 1}
    profile_to_idx = {'profile1': 0, 'profile2': 1}
    return interaction_matrix, X_features, user_to_idx, profile_to_idx

@pytest.fixture
def sample_profiles():
    return pd.DataFrame({
        '__id__': ['profile1', 'profile2'],
        'userName': ['Amani', 'Juma'],
        'country_match': [True, False],
        'language_match': [True, False],
        'goal_match': [True, False],
        'ml_score': [0.0, 0.0]
    })

def test_train_model(sample_data):
    interaction_matrix, X_features, _, _ = sample_data
    model, scaler = train_model(interaction_matrix, X_features)
    
    assert isinstance(model, CompatibilityModel), "Model should be a CompatibilityModel instance"
    assert isinstance(scaler, StandardScaler), "Scaler should be a StandardScaler instance"
    assert model.network[-1].out_features == 1, "Model should output a single value"

def test_predict_compatibility(sample_data, sample_profiles):
    interaction_matrix, X_features, user_to_idx, profile_to_idx = sample_data
    model, scaler = train_model(interaction_matrix, X_features)
    
    filtered_profiles = predict_compatibility(model, scaler, 'user1', sample_profiles, X_features, 
                                             user_to_idx, profile_to_idx)
    
    assert 'ml_score' in filtered_profiles.columns, "ml_score column should be present"
    assert 'final_score' in filtered_profiles.columns, "final_score column should be present"
    assert len(filtered_profiles) == 2, "Should return all profiles"
    assert filtered_profiles['final_score'].max() > 0, "Final scores should be positive"
