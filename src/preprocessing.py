import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import logging
from src.data_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(profiles, liked, matched):
    """
    Preprocess profiles and create interaction matrix.
    Returns: processed_profiles, interaction_matrix, X_features, user_to_idx, profile_to_idx, label_encoders, tfidf
    """
    config = load_config()
    categorical_cols = config['categorical_columns']
    tfidf_params = config['tfidf_params']
    keywords = config['keywords']

    try:
        # Label encoding for categorical columns
        label_encoders = {}
        for col in categorical_cols:
            if col in profiles.columns:
                le = LabelEncoder()
                unique_values = list(profiles[col].unique()) + ['unknown']
                le.fit(unique_values)
                profiles[col] = le.transform(profiles[col]).astype(np.int64)
                label_encoders[col] = le
            else:
                logger.warning(f"Column {col} not found in profiles")

        # Numeric and boolean preprocessing
        profiles['age'] = pd.to_numeric(profiles['age'], errors='coerce').fillna(0).astype(np.int64)
        for col in ['subscribed', 'subscribedEliteOne', 'subscribedEliteThree', 'subscribedEliteSix', 'subscribedEliteTwelve']:
            profiles[col] = profiles[col].map({True: 1, False: 0, 'unknown': 0}).astype(np.int64)

        # TF-IDF for aboutMe
        tfidf = TfidfVectorizer(**tfidf_params)
        tfidf_matrix = tfidf.fit_transform(profiles['aboutMe'])
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Keyword relevance
        profiles['keyword_score'] = profiles['aboutMe'].apply(
            lambda x: sum(1 for word in keywords if word.lower() in str(x).lower()) / len(keywords)
        ).astype(np.float64)

        # Interaction matrix
        user_ids = profiles['userId'].unique()
        profile_ids = profiles['__id__'].unique()
        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        profile_to_idx = {pid: idx for idx, pid in enumerate(profile_ids)}

        interactions = []
        for _, row in liked.iterrows():
            if row['userId'] in user_to_idx and row['__id__'] in profile_to_idx:
                interactions.append((user_to_idx[row['userId']], profile_to_idx[row['__id__']], 1))
        for _, row in matched.iterrows():
            if row['userId'] in user_to_idx and row['__id__'] in profile_to_idx:
                interactions.append((user_to_idx[row['userId']], profile_to_idx[row['__id__']], 2))

        if interactions:
            rows, cols, values = zip(*interactions)
            interaction_matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(profile_ids)))
        else:
            logger.warning("No interactions found")
            interaction_matrix = csr_matrix((len(user_ids), len(profile_ids)))

        # Combine features
        numeric_features = ['age'] + categorical_cols + ['subscribed', 'subscribedEliteOne', 
                                                        'subscribedEliteThree', 'subscribedEliteSix', 
                                                        'subscribedEliteTwelve', 'keyword_score']
        X_numeric = profiles[numeric_features]
        X_features = hstack([X_numeric.values.astype(np.float64), tfidf_matrix]).tocsr()
        logger.info(f"Feature matrix shape: {X_features.shape}")

        return (profiles, interaction_matrix, X_features, user_to_idx, profile_to_idx, label_encoders, tfidf)

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise