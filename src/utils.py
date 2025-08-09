import joblib
import torch
import os
import pandas as pd
import logging
from src.data_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_models(model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx, models_dir=None):
    """
    Save trained models and encoders to models_dir.
    """
    config = load_config()
    models_dir = models_dir or config['models_dir']
    
    try:
        os.makedirs(models_dir, exist_ok=True)
        torch.save(model, os.path.join(models_dir, "pytorch_model.pth"))
        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
        joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))
        joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        joblib.dump(user_to_idx, os.path.join(models_dir, "user_to_idx.pkl"))
        joblib.dump(profile_to_idx, os.path.join(models_dir, "profile_to_idx.pkl"))
        logger.info(f"Saved models to {models_dir}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise

def save_recommendations(top_matches, output_dir=None):
    """
    Save top matches to CSV.
    """
    config = load_config()
    output_dir = output_dir or config['data_dir']
    
    try:
        recommendations = top_matches[['__id__', 'userName', 'final_score']].copy()
        recommendations['reasons'] = top_matches.apply(
            lambda row: [
                "Country match" if row['country_match'] else None,
                "Language match" if row['language_match'] else None,
                "Relationship goals match" if row['goal_match'] else None,
                "High ML compatibility" if row['ml_score'] > 0.5 else None,
                "Subscribed user" if row['subscribed_score'] > 0 else None
            ], axis=1)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "recommendations.csv")
        recommendations.to_csv(output_path, index=False)
        logger.info(f"Saved recommendations to {output_path}")
    except Exception as e:
        logger.error(f"Error saving recommendations: {e}")
        raise