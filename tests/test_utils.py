import pytest
import pandas as pd
import os
import torch
from src.utils import save_models, save_recommendations
from src.recommender import CompatibilityModel

@pytest.fixture
def temp_dir(tmp_path):
    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    models_dir.mkdir()
    data_dir.mkdir()
    return str(models_dir), str(data_dir)

@pytest.fixture
def sample_model_and_encoders():
    model = CompatibilityModel(input_dim=5, hidden_dims=[64, 32])
    scaler = StandardScaler()
    label_encoders = {'country': LabelEncoder()}
    tfidf = TfidfVectorizer(max_features=50)
    user_to_idx = {'user1': 0}
    profile_to_idx = {'profile1': 0}
    return model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx

@pytest.fixture
def sample_top_matches():
    return pd.DataFrame({
        '__id__': ['profile1'],
        'userName': ['Amani'],
        'final_score': [0.9],
        'country_match': [True],
        'language_match': [True],
        'goal_match': [True],
        'ml_score': [0.8],
        'subscribed_score': [1]
    })

def test_save_models(temp_dir, sample_model_and_encoders):
    models_dir, _ = temp_dir
    model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx = sample_model_and_encoders
    
    save_models(model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx, models_dir=models_dir)
    
    assert os.path.exists(os.path.join(models_dir, "pytorch_model.pth")), "PyTorch model should be saved"
    assert os.path.exists(os.path.join(models_dir, "scaler.pkl")), "Scaler should be saved"
    assert os.path.exists(os.path.join(models_dir, "label_encoders.pkl")), "Label encoders should be saved"

def test_save_recommendations(temp_dir, sample_top_matches):
    _, data_dir = temp_dir
    save_recommendations(sample_top_matches, output_dir=data_dir)
    
    output_path = os.path.join(data_dir, "recommendations.csv")
    assert os.path.exists(output_path), "Recommendations CSV should be saved"
    df = pd.read_csv(output_path)
    assert 'reasons' in df.columns, "Reasons column should be present"
    assert df['final_score'].iloc[0] == 0.9, "Final score should match"