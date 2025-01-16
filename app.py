import streamlit as st
import time
from model import YoutubeTitlePredictor
from gemini_helper import GeminiTitleGenerator
from config import (
    YOUTUBE_API_KEY,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE
)

# Configure page settings
st.set_page_config(
    page_title="YouTube Title Generator",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS for animations and styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Custom title styling */
    .title {
        background: linear-gradient(45deg, #FF0000, #FF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        animation: slideDown 0.5s ease-out;
    }
    
    /* Card-like containers */
    .stApp > header {
        background-color: transparent;
    }
    
    .card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        animation: slideUp 0.5s ease-out;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #2b5876, #4e4376);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 5px;
        border: 2px solid #eee;
        padding: 10px;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #2b5876;
        box-shadow: 0 0 5px rgba(43, 88, 118, 0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideDown {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, #2b5876, #4e4376);
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_predictor():
    """Initialize the YouTube Title Predictor"""
    try:
        return YoutubeTitlePredictor(YOUTUBE_API_KEY)
    except Exception as e:
        st.error(f"Error initializing YouTube API: {str(e)}")
        st.stop()

def initialize_gemini():
    """Initialize the Gemini Title Generator"""
    try:
        return GeminiTitleGenerator()
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        st.stop()

def display_training_results(titles, history):
    """Display training results with animation"""
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success("✨ Model trained successfully!")
        
        # Animated progress display
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        
        # Sample titles with animation
        st.subheader("🎯 Sample Original Titles")
        for i, title in enumerate(titles[:5]):
            time.sleep(0.2)  # Creates a typing effect
            st.write(f"#{i+1}: {title}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Title and Introduction
    st.markdown('<h1 class="title">🎥 YouTube Title Generator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; animation: fadeIn 1s ease-in;'>
        Create engaging YouTube titles with AI-powered suggestions and Hinglish clickbait versions!
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictors
    youtube_predictor = initialize_predictor()
    gemini_generator = initialize_gemini()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    # Sidebar/Training Options (Column 1)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔧 Training Options")
        
        channel_id = st.text_input("YouTube Channel ID", placeholder="Enter channel ID...")
        channel_name = st.text_input("Channel Name", placeholder="Enter channel name...")
        
        with st.expander("⚙️ Advanced Options"):
            epochs = st.slider("Training Epochs", 10, 100, DEFAULT_EPOCHS)
            batch_size = st.slider("Batch Size", 16, 64, DEFAULT_BATCH_SIZE)
        
        train_button = st.button("🚀 Train Model")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Title Generation (Column 2)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✨ Generate New Titles")
        
        seed_text = st.text_input("Starting words", placeholder="Begin your title here...")
        num_words = st.slider("Words to generate", 3, 10, 6)
        generate_button = st.button("🎨 Generate Titles")
        
        # Handle Training
        if train_button and channel_id:
            try:
                with st.spinner("🧠 Training model... This may take several minutes."):
                    titles, history = youtube_predictor.train(
                        channel_id, 
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    youtube_predictor.save_model()
                    display_training_results(titles, history)
                    
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
        
        # Handle Generation
        if generate_button and seed_text:
            try:
                youtube_predictor.load_model()
                
                with st.spinner("🎯 Generating title..."):
                    generated_title = youtube_predictor.generate_title(
                        seed_text, 
                        next_words=num_words
                    )
                
                if channel_name:
                    st.markdown("### 🎨 Generated Titles")
                    
                    with st.spinner("✍️ Creating variations..."):
                        clickbait_titles = gemini_generator.generate_clickbait_titles(
                            generated_title,
                            channel_name
                        )
                        st.markdown(f"""
                        <div class="card" style="animation: slideUp 0.5s ease-out;">
                            {clickbait_titles}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("👉 Add a channel name to generate Hinglish clickbait titles!")
                    
            except FileNotFoundError:
                st.error("⚠️ Please train the model first!")
            except Exception as e:
                st.error(f"❌ Generation error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()