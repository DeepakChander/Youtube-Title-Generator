# # config.py
# import os
# from dotenv import load_dotenv


# # Load environment variables
# load_dotenv()

# # Get API key with proper error message
# YOUTUBE_API_KEY = 'AIzaSyBpT9KSdU6mGR1hlKpJulxshlineIARS_w'

# if not YOUTUBE_API_KEY:
#     raise ValueError("""
#     YouTube API key not found! Please:
#     1. Create a file named '.env' in your project directory
#     2. Add your YouTube API key like this: YOUTUBE_API_KEY=your_api_key_here
#     3. Make sure there are no spaces around the = sign
#     4. Restart the application
#     """)

# # Model Configuration
# MAX_TITLE_LENGTH = 50
# MAX_VIDEOS = 200
# EMBEDDING_DIM = 100
# DEFAULT_EPOCHS = 50
# DEFAULT_BATCH_SIZE = 32

# # Model Save Path
# MODEL_PATH = "saved_model"

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
YOUTUBE_API_KEY = 'AIzaSyBpT9KSdU6mGR1hlKpJulxshlineIARS_w'
GEMINI_API_KEY = 'AIzaSyD1A5pxBCdICYp8dDkVrRR41MvyhcAcm3M'

# Validate API keys
if not YOUTUBE_API_KEY:
    raise ValueError("""
    YouTube API key not found! Please add to your .env file:
    YOUTUBE_API_KEY=your_youtube_api_key
    """)

if not GEMINI_API_KEY:
    raise ValueError("""
    Gemini API key not found! Please add to your .env file:
    GEMINI_API_KEY=your_gemini_api_key
    """)

# Model Configuration
MAX_TITLE_LENGTH = 50
MAX_VIDEOS = 200
EMBEDDING_DIM = 100
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32

# Model Save Path
MODEL_PATH = "saved_model"