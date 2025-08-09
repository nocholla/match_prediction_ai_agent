import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import patch
from src.data_loader import load_data, load_config
from src.preprocessing import preprocess_data
from src.recommender import load_model_and_encoders
from ui.streamlit_app import main

@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a temporary data directory with mock CSV files."""
    data_path = tmp_path / "data"
    data_path.mkdir()
    
    profiles_data = pd.DataFrame({
        '__id__': [1, 2],
        'userId': ['user1', 'user2'],
        'userName': ['Alice', 'Bob'],
        'age': [25, 30],
        'country': ['Kenya', 'Nigeria'],
        'language': ['Swahili', 'English'],
        'aboutMe': ['Love soccer', 'Enjoy music'],
        'sex': ['Female', 'Male'],
        'seeking': ['Male', 'Female'],
        'relationshipGoals': ['Long-term', 'Casual'],
        'subscribed': [True, False],
        'subscribedEliteOne': [False, False],
        'subscribedEliteThree': [False, False],
        'subscribedEliteSix': [False, False],
        'subscribedEliteTwelve': [False, False]
    })
    profiles_data.to_csv(data_path / "Profiles.csv", index=False)
    
    for file in ['LikedUsers.csv', 'MatchedUsers.csv', 'BlockedUsers.csv', 
                 'DeclinedUsers.csv', 'DeletedUsers.csv', 'ReportedUsers.csv']:
        pd.DataFrame({'__id__': [], 'userId': []}).to_csv(data_path / file, index=False)
    
    return str(data_path)

@pytest.fixture
def mock_model_files(tmp_path):
    """Create mock model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    for file in ["matchmaking_model.pkl", "scaler.pkl", "label_encoders.pkl", 
                 "tfidf_vectorizer.pkl", "user_to_idx.pkl", "profile_to_idx.pkl"]:
        (model_dir / file).write_bytes(b"")
    
    return str(model_dir)

@patch("streamlit.text_input")
@patch("streamlit.slider")
@patch("streamlit.selectbox")
@patch("streamlit.text_area")
@patch("streamlit.button")
def test_main_with_existing_models(mock_button, mock_text_area, mock_selectbox, mock_slider, mock_text_input, mock_data_dir, mock_model_files):
    """Test main function when model files exist."""
    mock_text_input.side_effect = ["user123", "Kenya", "Swahili", "Long-term"]
    mock_slider.return_value = 25
    mock_selectbox.side_effect = ["Female", "Male"]
    mock_text_area.return_value = "Looking for true love and enjoy soccer!"
    mock_button.return_value = False  # Simulate button not clicked

    with patch("src.data_loader.load_config") as mock_load_config:
        mock_load_config.return_value = {
            'data': {
                'data_dir': mock_data_dir,
                'profiles_file': 'Profiles.csv',
                'liked_file': 'LikedUsers.csv',
                'matched_file': 'MatchedUsers.csv',
                'blocked_file': 'BlockedUsers.csv',
                'declined_file': 'DeclinedUsers.csv',
                'deleted_file': 'DeletedUsers.csv',
                'reported_file': 'ReportedUsers.csv',
                'required_columns': [
                    '__id__', 'userId', 'userName', 'age', 'country', 'language',
                    'aboutMe', 'sex', 'seeking', 'relationshipGoals', 'subscribed',
                    'subscribedEliteOne', 'subscribedEliteThree', 'subscribedEliteSix',
                    'subscribedEliteTwelve'
                ]
            },
            'model': {
                'models_dir': mock_model_files,
                'max_tfidf_features': 50
            },
            'preprocessing': {
                'categorical_columns': ['country', 'language', 'sex', 'seeking', 'relationshipGoals'],
                'tfidf_params': {'max_features': 50, 'stop_words': 'english', 'min_df': 1},
                'keywords': ['love', 'soul mate', 'relationship', 'partner', 'soccer', 'football']
            }
        }
        with patch("src.recommender.load_model_and_encoders") as mock_load_model:
            mock_load_model.return_value = (None, None, {}, None, {}, {})  # Mock model and encoders
            with patch("streamlit.error") as mock_error:
                main()
                mock_error.assert_not_called()  # Ensure no errors are raised

def test_main_without_models(mock_data_dir):
    """Test main function when model files do not exist."""
    with patch("src.data_loader.load_config") as mock_load_config:
        mock_load_config.return_value = {
            'data': {
                'data_dir': mock_data_dir,
                'profiles_file': 'Profiles.csv',
                'liked_file': 'LikedUsers.csv',
                'matched_file': 'MatchedUsers.csv',
                'blocked_file': 'BlockedUsers.csv',
                'declined_file': 'DeclinedUsers.csv',
                'deleted_file': 'DeletedUsers.csv',
                'reported_file': 'ReportedUsers.csv',
                'required_columns': [
                    '__id__', 'userId', 'userName', 'age', 'country', 'language',
                    'aboutMe', 'sex', 'seeking', 'relationshipGoals', 'subscribed',
                    'subscribedEliteOne', 'subscribedEliteThree', 'subscribedEliteSix',
                    'subscribedEliteTwelve'
                ]
            },
            'model': {
                'models_dir': 'nonexistent_models',
                'max_tfidf_features': 50
            },
            'preprocessing': {
                'categorical_columns': ['country', 'language', 'sex', 'seeking', 'relationshipGoals'],
                'tfidf_params': {'max_features': 50, 'stop_words': 'english', 'min_df': 1},
                'keywords': ['love', 'soul mate', 'relationship', 'partner', 'soccer', 'football']
            }
        }
        with patch("streamlit.error") as mock_error:
            main()
            mock_error.assert_not_called()  # Ensure no errors are raised