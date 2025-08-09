import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import os
import yaml
from src.data_loader import load_data, load_config

@pytest.fixture
def temp_csv_files(tmp_path):
    config = {
        'data_dir': str(tmp_path / "data"),
        'required_columns': ['__id__', 'userId', 'userName', 'age', 'country', 'language', 
                            'aboutMe', 'sex', 'seeking', 'relationshipGoals', 'subscribed',
                            'subscribedEliteOne', 'subscribedEliteThree', 'subscribedEliteSix',
                            'subscribedEliteTwelve'],
        'categorical_columns': ['country', 'language', 'sex', 'seeking', 'relationshipGoals'],
        'tfidf_params': {'max_features': 50, 'stop_words': 'english', 'min_df': 2},
        'keywords': ['love', 'soul mate', 'relationship', 'partner', 'soccer', 'football'],
        'model_params': {'hidden_dims': [64, 32]},
        'training_params': {'batch_size': 32, 'learning_rate': 0.001, 'epochs': 10},
        'models_dir': 'models'
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(tmp_path / "config.yaml", 'w') as f:
        yaml.safe_dump(config, f)

    profiles_data = {
        '__id__': ['profile1', 'profile2'],
        'userId': ['user1', 'user2'],
        'userName': ['Amani', 'Juma'],
        'age': [25, 30],
        'country': ['Kenya', 'Nigeria'],
        'language': ['Swahili', 'English'],
        'aboutMe': ['Love soccer', 'Seeking soul mate'],
        'sex': ['Female', 'Male'],
        'seeking': ['Male', 'Female'],
        'relationshipGoals': ['Long-term', 'Long-term'],
        'subscribed': [True, False],
        'subscribedEliteOne': [False, False],
        'subscribedEliteThree': [False, False],
        'subscribedEliteSix': [False, False],
        'subscribedEliteTwelve': [False, False]
    }
    liked_data = {'userId': ['user1'], '__id__': ['profile2']}
    matched_data = {'userId': ['user2'], '__id__': ['profile1']}
    blocked_data = {'__id__': ['profile3']}
    declined_data = {'__id__': ['profile4']}
    deleted_data = {'__id__': ['profile5']}
    reported_data = {'__id__': ['profile6']}

    pd.DataFrame(profiles_data).to_csv(data_dir / "Profiles.csv", index=False)
    pd.DataFrame(liked_data).to_csv(data_dir / "LikedUsers.csv", index=False)
    pd.DataFrame(matched_data).to_csv(data_dir / "MatchedUsers.csv", index=False)
    pd.DataFrame(blocked_data).to_csv(data_dir / "BlockedUsers.csv", index=False)
    pd.DataFrame(declined_data).to_csv(data_dir / "DeclinedUsers.csv", index=False)
    pd.DataFrame(deleted_data).to_csv(data_dir / "DeletedUsers.csv", index=False)
    pd.DataFrame(reported_data).to_csv(data_dir / "ReportedUsers.csv", index=False)
    
    return str(data_dir), str(tmp_path / "config.yaml")

def test_load_data_success(temp_csv_files):
    data_dir, _ = temp_csv_files
    profiles, liked, matched, blocked_ids, declined_ids, deleted_ids, reported_ids = load_data(data_dir=data_dir)
    
    assert profiles is not None, "Profiles DataFrame should not be None"
    assert liked is not None, "Liked DataFrame should not be None"
    assert matched is not None, "Matched DataFrame should not be None"
    assert len(blocked_ids) == 1, "Blocked IDs should contain one entry"
    assert len(declined_ids) == 1, "Declined IDs should contain one entry"
    assert len(deleted_ids) == 1, "Deleted IDs should contain one entry"
    assert len(reported_ids) == 1, "Reported IDs should contain one entry"
    
    assert profiles.shape[0] == 2, "Profiles should have 2 rows"
    assert '__id__' in profiles.columns, "Profiles should have __id__ column"
    assert 'userName' in profiles.columns, "Profiles should have userName column"
    assert liked.shape[0] == 1, "LikedUsers should have 1 row"
    assert matched.shape[0] == 1, "MatchedUsers should have 1 row"

def test_load_data_file_not_found(tmp_path):
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()
    profiles, liked, matched, blocked_ids, declined_ids, deleted_ids, reported_ids = load_data(data_dir=str(empty_dir))
    
    assert profiles is None, "Profiles should be None when files are missing"
    assert liked is None, "Liked should be None when files are missing"
    assert matched is None, "Matched should be None when files are missing"
    assert blocked_ids is None, "Blocked IDs should be None when files are missing"

def test_load_data_missing_columns(temp_csv_files, tmp_path):
    data_dir, _ = temp_csv_files
    invalid_data = pd.DataFrame({'__id__': ['profile1'], 'userId': ['user1']})
    invalid_data.to_csv(os.path.join(data_dir, "Profiles.csv"), index=False)
    
    with pytest.raises(ValueError, match="Missing columns in Profiles.csv"):
        load_data(data_dir=data_dir)