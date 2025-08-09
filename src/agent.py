import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from src.data_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_user_profile(user_profile):
    """Validate user profile inputs."""
    required_keys = ['userId', 'age', 'sex', 'seeking', 'country', 'language', 'relationshipGoals', 'aboutMe']
    for key in required_keys:
        if key not in user_profile:
            logger.error(f"Missing required user profile key: {key}")
            raise ValueError(f"Missing required user profile key: {key}")
    if not isinstance(user_profile['age'], (int, float)) or user_profile['age'] < 18 or user_profile['age'] > 70:
        logger.error(f"Invalid age: {user_profile['age']}")
        raise ValueError("Age must be a number between 18 and 70")
    if user_profile['sex'] not in ['Female', 'Male', 'unknown']:
        logger.error(f"Invalid sex: {user_profile['sex']}")
        raise ValueError("Sex must be 'Female', 'Male', or 'unknown'")
    if user_profile['seeking'] not in ['Female', 'Male', 'unknown']:
        logger.error(f"Invalid seeking: {user_profile['seeking']}")
        raise ValueError("Seeking must be 'Female', 'Male', or 'unknown'")
    if len(user_profile['country']) > 100:
        logger.error(f"Country too long: {user_profile['country']}")
        raise ValueError("Country must be less than 100 characters")
    if len(user_profile['language']) > 100:
        logger.error(f"Language too long: {user_profile['language']}")
        raise ValueError("Language must be less than 100 characters")
    if len(user_profile['relationshipGoals']) > 100:
        logger.error(f"Relationship goals too long: {user_profile['relationshipGoals']}")
        raise ValueError("Relationship Goals must be less than 100 characters")
    if len(user_profile['aboutMe']) > 1000:
        logger.error(f"About Me too long: {user_profile['aboutMe']}")
        raise ValueError("About Me must be less than 1000 characters")

def apply_rules(profiles, user_profile, blocked_ids, declined_ids, deleted_ids, reported_ids):
    """
    Apply rule-based filtering to profiles.
    Returns: filtered profiles with match scores
    """
    try:
        validate_user_profile(user_profile)
        filtered = profiles.copy()
        
        # Exclude blocked, declined, deleted, reported users
        excluded_ids = set(blocked_ids).union(declined_ids, deleted_ids, reported_ids)
        filtered = filtered[~filtered['__id__'].isin(excluded_ids)]
        
        # Match seeking and sex
        if user_profile.get('seeking') and user_profile.get('sex'):
            filtered = filtered[filtered['sex'] == user_profile['seeking']]
            filtered = filtered[filtered['seeking'] == user_profile['sex']]
        
        # Age range (±5 years)
        if user_profile.get('age'):
            age = float(user_profile['age'])
            filtered = filtered[filtered['age'].between(age - 5, age + 5)]
        
        # Prefer same country, language, or relationship goals
        filtered['country_match'] = filtered['country'] == user_profile.get('country', 'unknown')
        filtered['language_match'] = filtered['language'] == user_profile.get('language', 'unknown')
        filtered['goal_match'] = filtered['relationshipGoals'] == user_profile.get('relationshipGoals', 'unknown')
        
        # Prioritize subscribed users
        filtered['subscribed_score'] = filtered[['subscribed', 'subscribedEliteOne', 
                                                'subscribedEliteThree', 'subscribedEliteSix', 
                                                'subscribedEliteTwelve']].sum(axis=1)
        
        logger.info(f"Filtered to {len(filtered)} profiles after applying rules")
        return filtered
    
    except Exception as e:
        logger.error(f"Error in apply_rules: {e}")
        raise

def encode_user_profile(user_profile, label_encoders, tfidf):
    """
    Encode user profile for ML prediction.
    Returns: encoded user features
    """
    try:
        validate_user_profile(user_profile)
        config = load_config()
        keywords = config['keywords']
        
        user_features = []
        for col in ['country', 'language', 'sex', 'seeking', 'relationshipGoals']:
            if col in label_encoders and user_profile[col] != 'unknown':
                try:
                    user_features.append(label_encoders[col].transform([user_profile[col]])[0])
                except:
                    user_features.append(label_encoders[col].transform(['unknown'])[0])
            else:
                user_features.append(0)
        
        user_features.extend([
            user_profile['age'],
            0, 0, 0, 0, 0,  # Subscribed flags
            sum(1 for word in keywords if word.lower() in user_profile['aboutMe'].lower()) / len(keywords)
        ])
        
        tfidf_vec = tfidf.transform([user_profile['aboutMe']]).toarray().flatten()
        encoded_features = np.concatenate([user_features, tfidf_vec])
        logger.info(f"Encoded user profile with {len(encoded_features)} features")
        return encoded_features
    
    except Exception as e:
        logger.error(f"Error in encode_user_profile: {e}")
        raise