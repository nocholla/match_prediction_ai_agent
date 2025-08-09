import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.agent import apply_rules, encode_user_profile, validate_user_profile
from src.data_loader import load_config

@pytest.fixture
def sample_profiles():
    data = {
        '__id__': ['profile1', 'profile2', 'profile3'],
        'userId': ['user1', 'user2', 'user3'],
        'userName': ['Amani', 'Juma', 'Fatima'],
        'age': [25, 30, 28],
        'country': ['Kenya', 'Nigeria', 'Kenya'],
        'language': ['Swahili', 'English', 'Swahili'],
        'aboutMe': ['Love soccer', 'Seeking soul mate', 'Enjoy football'],
        'sex': ['Female', 'Male', 'Female'],
        'seeking': ['Male', 'Female', 'Male'],
        'relationshipGoals': ['Long-term', 'Long-term', 'Casual'],
        'subscribed': [1, 0, 1],
        'subscribedEliteOne': [0, 0, 0],
        'subscribedEliteThree': [0, 0, 0],
        'subscribedEliteSix': [0, 0, 0],
        'subscribedEliteTwelve': [0, 0, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_user_profile():
    return {
        'userId': 'user123',
        'age': 27,
        'sex': 'Male',
        'seeking': 'Female',
        'country': 'Kenya',
        'language': 'Swahili',
        'relationshipGoals': 'Long-term',
        'aboutMe': 'Looking for true love and enjoy soccer'
    }

@pytest.fixture
def sample_label_encoders():
    config = load_config()
    encoders = {}
    for col in config['categorical_columns']:
        le = LabelEncoder()
        le.fit([col, 'unknown'])
        encoders[col] = le
    return encoders

@pytest.fixture
def sample_tfidf():
    config = load_config()
    tfidf = TfidfVectorizer(**config['tfidf_params'])
    tfidf.fit(['Love soccer', 'Seeking soul mate', 'Enjoy football'])
    return tfidf

def test_apply_rules(sample_profiles, sample_user_profile):
    blocked_ids = ['profile3']
    declined_ids = []
    deleted_ids = []
    reported_ids = []
    
    filtered = apply_rules(sample_profiles, sample_user_profile, blocked_ids, declined_ids, deleted_ids, reported_ids)
    
    assert filtered.shape[0] == 1, "Should filter to 1 profile (profile1: Female, seeking Male, age 25, Kenya, Long-term)"
    assert filtered['__id__'].iloc[0] == 'profile1', "Profile1 should be selected"
    assert filtered['country_match'].iloc[0] == True, "Country should match"
    assert filtered['language_match'].iloc[0] == True, "Language should match"
    assert filtered['goal_match'].iloc[0] == True, "Relationship goals should match"
    assert filtered['subscribed_score'].iloc[0] == 1, "Subscribed score should be 1"

def test_encode_user_profile(sample_user_profile, sample_label_encoders, sample_tfidf):
    user_features = encode_user_profile(sample_user_profile, sample_label_encoders, sample_tfidf)
    
    assert len(user_features) >= 12 + 50, "Features should include numeric (12) + TF-IDF (50)"
    assert user_features[5] == 27, "Age should be 27"
    assert user_features[6:11].sum() == 0, "Subscribed flags should be 0"
    assert user_features[11] > 0, "Keyword score should be positive due to 'love' and 'soccer'"
    assert user_features[12:].sum() > 0, "TF-IDF features should have non-zero values for 'love' and 'soccer'"

def test_validate_user_profile_invalid_age(sample_user_profile):
    invalid_profile = sample_user_profile.copy()
    invalid_profile['age'] = 15
    with pytest.raises(ValueError, match="Age must be a number between 18 and 70"):
        validate_user_profile(invalid_profile)

def test_validate_user_profile_invalid_sex(sample_user_profile):
    invalid_profile = sample_user_profile.copy()
    invalid_profile['sex'] = 'Other'
    with pytest.raises(ValueError, match="Sex must be 'Female', 'Male', or 'unknown'"):
        validate_user_profile(invalid_profile)