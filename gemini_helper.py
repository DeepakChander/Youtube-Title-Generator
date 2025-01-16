import google.generativeai as genai
from config import GEMINI_API_KEY

class GeminiTitleGenerator:
    def __init__(self):
        self.setup_gemini()
        
    def setup_gemini(self):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini API: {str(e)}")
    
    def generate_clickbait_titles(self, title, channel_name):
        """Generate clickbait titles in Hinglish"""
        try:
            prompt = f"""
            Based on this YouTube title: "{title}"
            For YouTube channel: "{channel_name}"
            Generate 3 catchy clickbait titles in English.
            Rules:
            1. Make them engaging and attention-grabbing
            2. Keep them relevant to the original title
            3. Use common Hinglish phrases
            4. Include emojis where appropriate
            5. Format: Number each title (1., 2., 3.)
            6. And all the generated words should be witten in English alphabets.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            raise Exception(f"Error generating clickbait titles: {str(e)}")