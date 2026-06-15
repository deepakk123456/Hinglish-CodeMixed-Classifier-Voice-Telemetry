import re
import string
import nltk
import emoji

class HinglishPreprocessor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def remove_noise(self, text: str) -> str:
        """Removes URLs, HTML tags, and social media user tags."""
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'@\w+', '', text)
        return text

    def handle_emojis(self, text: str) -> str:
        """Converts emojis into readable text descriptions."""
        demojized = emoji.demojize(text, delimiters=(" ", " "))
        cleaned_emoji = demojized.replace("_", " ").replace(":", "")
        return cleaned_emoji

    def reduce_lengthening(self, text: str) -> str:
        """Reduces repeated characters used for social media emphasis (e.g., 'bhaaiii' -> 'bhaaii')."""
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

    def deep_clean(self, text: str) -> str:
        """Sequential preprocessing pipeline."""
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = self.remove_noise(text)
        text = self.handle_emojis(text)
        # FIX: Changed text.reduce_lengthening to self.reduce_lengthening
        text = self.reduce_lengthening(text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Self-test block to verify functionality
if __name__ == "__main__":
    preprocessor = HinglishPreprocessor()
    chaotic_input = "Bhaaiiiiii!!! Check out this link https://coolstuff.com... Aaj mood bahut kharaab hai 😡😢 @test_user"
    
    print("--- RAW USER INPUT ---")
    print(chaotic_input)
    
    cleaned_output = preprocessor.deep_clean(chaotic_input)
    
    print("\n--- PREPROCESSED OUTPUT ---")
    print(cleaned_output)