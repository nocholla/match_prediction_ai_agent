import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from src.data_loader import load_data, load_config
from src.preprocessing import preprocess_data
from src.recommender import train_model, predict_compatibility, load_model_and_scaler
from src.agent import apply_rules, encode_user_profile
from src.utils import save_models, save_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_args(args):
    """Validate CLI arguments."""
    try:
        if not args.user_id or len(args.user_id) > 50:
            raise ValueError("User ID must be non-empty and less than 50 characters")
        if args.age < 18 or args.age > 70:
            raise ValueError("Age must be between 18 and 70")
        if args.sex not in ["Female", "Male", "unknown"]:
            raise ValueError("Sex must be 'Female', 'Male', or 'unknown'")
        if args.seeking not in ["Female", "Male", "unknown"]:
            raise ValueError("Seeking must be 'Female', 'Male', or 'unknown'")
        if len(args.country) > 100:
            raise ValueError("Country must be less than 100 characters")
        if len(args.language) > 100:
            raise ValueError("Language must be less than 100 characters")
        if len(args.relationship_goals) > 100:
            raise ValueError("Relationship Goals must be less than 100 characters")
        if len(args.about_me) > 1000:
            raise ValueError("About Me must be less than 1000 characters")
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        print(f"Error: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Africa Love Match - Matchmaking AI Agent CLI")
    parser.add_argument("--user_id", default="user123", help="User ID")
    parser.add_argument("--age", type=int, default=25, help="Age (18-70)")
    parser.add_argument("--sex", default="unknown", choices=["Female", "Male", "unknown"], help="Sex")
    parser.add_argument("--seeking", default="unknown", choices=["Male", "Female", "unknown"], help="Seeking")
    parser.add_argument("--country", default="unknown", help="Country")
    parser.add_argument("--language", default="unknown", help="Language")
    parser.add_argument("--relationship_goals", default="unknown", help="Relationship Goals")
    parser.add_argument("--about_me", default="Looking for true love and enjoy soccer", help="About Me")

    args = parser.parse_args()
    if not validate_args(args):
        return

    config = load_config()
    user_profile = {
        'userId': args.user_id,
        'age': args.age,
        'sex': args.sex,
        'seeking': args.seeking,
        'country': args.country,
        'language': args.language,
        'relationshipGoals': args.relationship_goals,
        'aboutMe': args.about_me
    }

    # Load data
    (profiles, liked, matched, blocked_ids, declined_ids, deleted_ids, reported_ids) = load_data(data_dir=config['data_dir'])
    if profiles is None:
        print("Error: Failed to load data. Check logs for details.")
        return

    # Preprocess data
    try:
        (profiles, interaction_matrix, X_features, user_to_idx, profile_to_idx, 
         label_encoders, tfidf) = preprocess_data(profiles, liked, matched)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        print("Error: Preprocessing failed. Check logs for details.")
        return

    # Load or train model
    model, scaler = load_model_and_scaler(config['models_dir'])
    if model is None or scaler is None:
        logger.info("Training new PyTorch model")
        try:
            model, scaler = train_model(interaction_matrix, X_features)
            save_models(model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx, 
                        models_dir=config['models_dir'])
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            print("Error: Model training failed. Check logs for details.")
            return

    # Apply rules
    try:
        filtered_profiles = apply_rules(profiles, user_profile, blocked_ids, declined_ids, 
                                       deleted_ids, reported_ids)
    except Exception as e:
        logger.error(f"Rule application failed: {e}")
        print("Error: Rule application failed. Check logs for details.")
        return
    
    if filtered_profiles.empty:
        print("Error: No compatible profiles found after rule-based filtering.")
        logger.warning("No compatible profiles found after rule-based filtering")
        return

    # Encode user profile
    try:
        user_features = encode_user_profile(user_profile, label_encoders, tfidf)
    except Exception as e:
        logger.error(f"User profile encoding failed: {e}")
        print("Error: User profile encoding failed. Check logs for details.")
        return

    # Predict compatibility
    try:
        filtered_profiles = predict_compatibility(model, scaler, user_profile['userId'], 
                                                filtered_profiles, X_features, 
                                                user_to_idx, profile_to_idx)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print("Error: Prediction failed. Check logs for details.")
        return
    
    # Top 5 matches
    top_matches = filtered_profiles.sort_values('final_score', ascending=False).head(5)
    
    # Save recommendations
    try:
        save_recommendations(top_matches, output_dir=config['data_dir'])
    except Exception as e:
        logger.error(f"Failed to save recommendations: {e}")
        print("Error: Failed to save recommendations. Check logs for details.")
        return

    # Display results
    print("\nTop Compatible Profiles:")
    for _, row in top_matches.iterrows():
        print(f"Profile ID: {row['__id__']}")
        print(f"User Name: {row['userName']}")
        print(f"Age: {row['age']}")
        print(f"Country: {row['country']}")
        print(f"About Me: {row['aboutMe']}")
        print(f"Compatibility Score: {row['final_score']:.2f}")
        print("Reasons:")
        if row['country_match']:
            print("- Matches your country")
        if row['language_match']:
            print("- Matches your language")
        if row['goal_match']:
            print("- Matches your relationship goals")
        if row['ml_score'] > 0.5:
            print("- High AI-predicted compatibility")
        if row['subscribed_score'] > 0:
            print("- Subscribed user")
        if 'soccer' in row['aboutMe'].lower():
            print("- Soccer enthusiast (Africa Soccer Kings compatible)")
        print("---")

if __name__ == "__main__":
    main()