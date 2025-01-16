import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
import os
from config import (
    MAX_TITLE_LENGTH, 
    MAX_VIDEOS, 
    EMBEDDING_DIM,
    MODEL_PATH
)

class YoutubeTitlePredictor:
    def __init__(self, api_key):
        """Initialize the YouTube Title Predictor"""
        if not api_key:
            raise ValueError("API key is required")
            
        try:
            self.youtube = build('youtube', 'v3', 
                               developerKey=api_key,
                               static_discovery=False)
            self.tokenizer = Tokenizer()
            self.max_title_length = MAX_TITLE_LENGTH
            self.model = None
            self.max_sequence_len = None
            self._create_model_directory()
        except Exception as e:
            raise Exception(f"Failed to initialize YouTube API: {str(e)}")
    
    def _create_model_directory(self):
        """Create directory for saving model if it doesn't exist"""
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

    def get_channel_id_from_name(self, channel_name):
        """Fetch the channel ID from the channel name"""
        try:
            search_response = self.youtube.search().list(
                part="snippet",
                q=channel_name,
                type="channel",
                maxResults=1
            ).execute()

            if not search_response.get('items'):
                raise ValueError(f"Channel not found: {channel_name}")

            channel_id = search_response['items'][0]['snippet']['channelId']
            return channel_id

        except HttpError as e:
            raise Exception(f"YouTube API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching channel ID: {str(e)}")

    def get_channel_videos(self, channel_name):
        """Fetch video titles from a YouTube channel using its name."""
        try:
            # Search for the channel by name
            search_response = self.youtube.search().list(
                q=channel_name,
                type="channel",
                part="id",
                maxResults=1
            ).execute()

            if not search_response.get('items'):
                raise ValueError(f"Channel '{channel_name}' not found.")

            channel_id = search_response['items'][0]['id']['channelId']

            # Fetch channel's uploads playlist ID
            channel_response = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id
            ).execute()

            if not channel_response.get('items'):
                raise ValueError(f"Channel '{channel_name}' does not have any videos.")

            playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            # Fetch video titles from the playlist
            videos = []
            next_page_token = None

            while True:
                playlist_response = self.youtube.playlistItems().list(
                    part="snippet",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()

                for item in playlist_response['items']:
                    if len(videos) >= MAX_VIDEOS:
                        return videos
                    title = item['snippet']['title']
                    if not any(skip in title.lower() for skip in ['#shorts', '(live)', 'premiere']):
                        videos.append(title)

                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token or len(videos) >= MAX_VIDEOS:
                    break

            return videos

        except HttpError as e:
            raise Exception(f"YouTube API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching videos: {str(e)}")


    def prepare_sequences(self, titles):
        """Prepare sequences for training"""
        try:
            # Fit tokenizer
            self.tokenizer.fit_on_texts(titles)
            total_words = len(self.tokenizer.word_index) + 1
            
            # Create sequences
            input_sequences = []
            for title in titles:
                token_list = self.tokenizer.texts_to_sequences([title])[0]
                for i in range(1, len(token_list)):
                    n_gram_sequence = token_list[:i+1]
                    input_sequences.append(n_gram_sequence)
            
            if not input_sequences:
                raise ValueError("No valid sequences created from titles")
            
            # Pad sequences
            self.max_sequence_len = max([len(x) for x in input_sequences])
            input_sequences = pad_sequences(input_sequences, 
                                         maxlen=self.max_sequence_len,
                                         padding='pre')
            
            # Split into input and output
            X = input_sequences[:, :-1]
            y = input_sequences[:, -1]
            
            return X, y, total_words
            
        except Exception as e:
            raise Exception(f"Error preparing sequences: {str(e)}")

    def build_model(self, total_words):
        """Build the LSTM model"""
        try:
            self.model = Sequential([ 
                Embedding(total_words, EMBEDDING_DIM, 
                         input_length=self.max_sequence_len-1),
                LSTM(150, return_sequences=True),
                Dropout(0.2),
                LSTM(100),
                Dropout(0.2),
                Dense(total_words, activation='softmax')
            ])
            
            self.model.compile(loss='sparse_categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
            
            return self.model
            
        except Exception as e:
            raise Exception(f"Error building model: {str(e)}")

    def train(self, channel_name, epochs=50, batch_size=32):
        """Train the model on channel's video titles"""
        try:
            # Fetch and prepare data
            titles = self.get_channel_videos(channel_name)
            if len(titles) < 10:
                raise ValueError("Not enough videos to train (minimum 10 required)")
            
            X, y, total_words = self.prepare_sequences(titles)
            
            # Build and train model
            self.build_model(total_words)
            history = self.model.fit(X, y,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   validation_split=0.1,
                                   verbose=1)
            
            return titles, history
            
        except Exception as e:
            raise Exception(f"Training error: {str(e)}")

    def generate_title(self, seed_text, next_words=6):
        """Generate a new title based on seed text"""
        if not self.model or not self.tokenizer:
            raise Exception("Model not trained or loaded")

        if not seed_text.strip():
            return "Error: Seed text cannot be empty"

        try:
            generated_text = seed_text

            for _ in range(next_words):
                # Tokenize and pad sequence
                token_list = self.tokenizer.texts_to_sequences([generated_text])[0]
                token_list = pad_sequences([token_list],
                                        maxlen=self.max_sequence_len-1,
                                        padding='pre')

                # Predict next word
                predicted = self.model.predict(token_list, verbose=0)
                predicted = np.argmax(predicted, axis=-1)

                # Convert prediction to word
                output_word = ""
                for word, index in self.tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break

                if not output_word:
                    break  # Stop generating if no word is found

                generated_text += " " + output_word

            return generated_text

        except Exception as e:
            return f"Error generating title: {str(e)}"

    def save_model(self):
        """Save the model and tokenizer"""
        try:
            if not self.model:
                raise ValueError("No model to save")
                
            self.model.save(f"{MODEL_PATH}/model.h5")
            with open(f"{MODEL_PATH}/tokenizer.pkl", "wb") as f:
                pickle.dump(self.tokenizer, f)
            with open(f"{MODEL_PATH}/max_sequence_len.pkl", "wb") as f:
                pickle.dump(self.max_sequence_len, f)
                
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")

    def load_model(self):
        """Load the model and tokenizer"""
        try:
            if not os.path.exists(f"{MODEL_PATH}/model.h5"):
                raise FileNotFoundError("No saved model found")
                
            self.model = load_model(f"{MODEL_PATH}/model.h5")
            with open(f"{MODEL_PATH}/tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)
            with open(f"{MODEL_PATH}/max_sequence_len.pkl", "rb") as f:
                self.max_sequence_len = pickle.load(f)
                
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
