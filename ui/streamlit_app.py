import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import time
import logging
from src.data_loader import load_data, load_config
from src.preprocessing import preprocess_data
from src.recommender import train_model, predict_compatibility, load_model_and_scaler
from src.agent import apply_rules, encode_user_profile
from src.utils import save_models, save_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_resource
def cached_load_data():
    config = load_config()
    return load_data(data_dir=config['data_dir'])

@st.cache_resource
def cached_preprocess_data(profiles, liked, matched):
    return preprocess_data(profiles, liked, matched)

@st.cache_resource
def cached_train_model(_interaction_matrix, _X_features):
    return train_model(_interaction_matrix, _X_features)

def validate_input(user_id, age, sex, seeking, country, language, relationship_goals, about_me):
    """Validate Streamlit input."""
    try:
        if not user_id or len(user_id) > 50:
            raise ValueError("User ID must be non-empty and less than 50 characters")
        if not isinstance(age, int) or age < 18 or age > 70:
            raise ValueError("Age must be between 18 and 70")
        if sex not in ["Female", "Male", "unknown"]:
            raise ValueError("Sex must be 'Female', 'Male', or 'unknown'")
        if seeking not in ["Female", "Male", "unknown"]:
            raise ValueError("Seeking must be 'Female', 'Male', or 'unknown'")
        if len(country) > 100:
            raise ValueError("Country must be less than 100 characters")
        if len(language) > 100:
            raise ValueError("Language must be less than 100 characters")
        if len(relationship_goals) > 100:
            raise ValueError("Relationship Goals must be less than 100 characters")
        if len(about_me) > 1000:
            raise ValueError("About Me must be less than 1000 characters")
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        st.error(str(e))
        return False
    return True

def main():
    st.title("👫 Match Prediction AI Agent")
    st.markdown("""
    Enter a user ID or profile details to get high-compatibility profile suggestions.
    Compatible with Africa Soccer Kings via soccer-related keywords.
    """)

    config = load_config()
    start_time = time.time()

    # Load data
    (profiles, liked, matched, blocked_ids, declined_ids, deleted_ids, reported_ids) = cached_load_data()
    if profiles is None:
        st.error("Failed to load data. Check logs for details.")
        st.stop()
    st.write(f"Data Loading Time: {time.time() - start_time:.2f} seconds")

    # Preprocess data
    try:
        (profiles, interaction_matrix, X_features, user_to_idx, profile_to_idx, 
         label_encoders, tfidf) = cached_preprocess_data(profiles, liked, matched)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        st.error("Preprocessing failed. Check logs for details.")
        st.stop()

    # Load or train model
    model, scaler = load_model_and_scaler(config['models_dir'])
    if model is None or scaler is None:
        logger.info("Training new PyTorch model")
        try:
            model, scaler = cached_train_model(interaction_matrix, X_features)
            save_models(model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx, 
                        models_dir=config['models_dir'])
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            st.error("Model training failed. Check logs for details.")
            st.stop()

    # User input
    user_id_input = st.text_input("User ID", "user123")
    age_input = st.slider("Age", 18, 70, 25)
    sex_input = st.selectbox("Sex", ["Female", "Male", "unknown"])
    seeking_input = st.selectbox("Seeking", ["Male", "Female", "unknown"])
    country_input = st.text_input("Country", "unknown")
    language_input = st.text_input("Language", "unknown")
    relationship_goals_input = st.text_input("Relationship Goals", "unknown")
    about_me_input = st.text_area("About Me", "Looking for true love and enjoy soccer!")

    if st.button("Find Matches"):
        if not validate_input(user_id_input, age_input, sex_input, seeking_input, country_input, 
                             language_input, relationship_goals_input, about_me_input):
            return

        user_profile = {
            'userId': user_id_input,
            'age': age_input,
            'sex': sex_input,
            'seeking': seeking_input,
            'country': country_input,
            'language': language_input,
            'relationshipGoals': relationship_goals_input,
            'aboutMe': about_me_input
        }
        
        # Apply rules
        try:
            filtered_profiles = apply_rules(profiles, user_profile, blocked_ids, declined_ids, 
                                           deleted_ids, reported_ids)
        except Exception as e:
            logger.error(f"Rule application failed: {e}")
            st.error("Rule application failed. Check logs for details.")
            return
        
        if filtered_profiles.empty:
            st.error("No compatible profiles found after rule-based filtering.")
            logger.warning("No compatible profiles found after rule-based filtering")
            return

        # Encode user profile
        try:
            user_features = encode_user_profile(user_profile, label_encoders, tfidf)
        except Exception as e:
            logger.error(f"User profile encoding failed: {e}")
            st.error("User profile encoding failed. Check logs for details.")
            return

        # Predict compatibility
        try:
            filtered_profiles = predict_compatibility(model, scaler, user_id_input, 
                                                    filtered_profiles, X_features, 
                                                    user_to_idx, profile_to_idx)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            st.error("Prediction failed. Check logs for details.")
            return
        
        # Top 5 matches
        top_matches = filtered_profiles.sort_values('final_score', ascending=False).head(5)
        
        # Save recommendations
        try:
            save_recommendations(top_matches, output_dir=config['data_dir'])
        except Exception as e:
            logger.error(f"Failed to save recommendations: {e}")
            st.error("Failed to save recommendations. Check logs for details.")
            return
        
        st.subheader("Top Compatible Profiles:")
        for _, row in top_matches.iterrows():
            st.write(f"Profile ID: {row['__id__']}")
            st.write(f"User Name: {row['userName']}")
            st.write(f"Age: {row['age']}")
            st.write(f"Country: {row['country']}")
            st.write(f"About Me: {row['aboutMe']}")
            st.write(f"Compatibility Score: {row['final_score']:.2f}")
            st.write("Reasons:")
            if row['country_match']:
                st.write("- Matches your country")
            if row['language_match']:
                st.write("- Matches your language")
            if row['goal_match']:
                st.write("- Matches your relationship goals")
            if row['ml_score'] > 0.5:
                st.write("- High AI-predicted compatibility")
            if row['subscribed_score'] > 0:
                st.write("- Subscribed user")
            if 'soccer' in row['aboutMe'].lower():
                st.write("- Soccer enthusiast (Africa Soccer Kings compatible)")
            st.write("---")

if __name__ == "__main__":
    main()