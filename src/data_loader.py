import pandas as pd
import os
import logging
import yaml
import time
from jsonschema import validate, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise

def validate_csv_schema(df, expected_cols, file_name):
    """Validate CSV file schema."""
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in {file_name}: {missing_cols}")
        raise ValueError(f"Missing columns in {file_name}: {missing_cols}")

def load_data(data_dir=None):
    """
    Load and merge CSV datasets.
    Returns: profiles_df, liked_df, matched_df, blocked_ids, declined_ids, deleted_ids, reported_ids
    """
    start_time = time.time()
    config = load_config()
    data_dir = data_dir or config['data_dir']
    required_cols = config['required_columns']

    try:
        profiles = pd.read_csv(
            os.path.join(data_dir, "Profiles.csv"),
            usecols=[c for c in required_cols if c in pd.read_csv(os.path.join(data_dir, "Profiles.csv"), nrows=1).columns]
        )
        validate_csv_schema(profiles, required_cols, "Profiles.csv")
        
        liked = pd.read_csv(os.path.join(data_dir, "LikedUsers.csv"), usecols=['userId', '__id__'])
        validate_csv_schema(liked, ['userId', '__id__'], "LikedUsers.csv")
        
        matched = pd.read_csv(os.path.join(data_dir, "MatchedUsers.csv"), usecols=['userId', '__id__'])
        validate_csv_schema(matched, ['userId', '__id__'], "MatchedUsers.csv")
        
        blocked = pd.read_csv(os.path.join(data_dir, "BlockedUsers.csv"), usecols=['__id__'])
        validate_csv_schema(blocked, ['__id__'], "BlockedUsers.csv")
        
        declined = pd.read_csv(os.path.join(data_dir, "DeclinedUsers.csv"), usecols=['__id__'])
        validate_csv_schema(declined, ['__id__'], "DeclinedUsers.csv")
        
        deleted = pd.read_csv(os.path.join(data_dir, "DeletedUsers.csv"), usecols=['__id__'])
        validate_csv_schema(deleted, ['__id__'], "DeletedUsers.csv")
        
        reported = pd.read_csv(os.path.join(data_dir, "ReportedUsers.csv"), usecols=['__id__'])
        validate_csv_schema(reported, ['__id__'], "ReportedUsers.csv")
        
        profiles = profiles.drop_duplicates().fillna("unknown")
        logger.info(f"Loaded {len(profiles)} profiles, {len(liked)} liked, {len(matched)} matched")
        logger.info(f"Data Loading Time: {time.time() - start_time:.2f} seconds")
        return (profiles, liked, matched, 
                blocked['__id__'].tolist(), declined['__id__'].tolist(), 
                deleted['__id__'].tolist(), reported['__id__'].tolist())
    
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        return None, None, None, None, None, None, None
    except ValueError as e:
        logger.error(f"Schema validation error: {e}")
        return None, None, None, None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        return None, None, None, None, None, None, None