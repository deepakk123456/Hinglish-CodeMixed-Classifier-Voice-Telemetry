import os
import torch
from transformers import pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class NextGenHinglishEngine:
    def __init__(self):
        print("🚀 Initializing Next-Gen Hinglish Sentiment Engine...")
        
        # TIP: Aap "l3ube/hinglish-sentiment-bert" ya "cardiffnlp/twitter-roberta-base-sentiment-latest" try kar sakte hain.
        # Twitter-RoBERTa Hinglish ke slang aur mix language ko default se kahin behtar samajhta hai.
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # GPU check agar system mein available ho toh performance boost ke liye
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            # return_all_scores=True se humein positive, negative, neutral teeno ka breakdown milega
            self.nlp_pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model_name, 
                tokenizer=self.model_name,
                device=self.device,
                top_k=None # Saare labels ke scores return karega
            )
            print(f"🔥 Neural Matrix Active on {'GPU' if self.device == 0 else 'CPU'}!")
        except Exception as e:
            print(f"⚠️ Initialization error: {e}")
            self.nlp_pipeline = None

    def predict_sentiment(self, text: str):
        if not self.nlp_pipeline:
            return "neutral", 0.50
        
        try:
            # Text ko clean ya normalize karne ki ab zarurat nahi, Transformer handle kar lega
            pipeline_output = self.nlp_pipeline(text)[0]
            
            # Outputs ko dictionary format mein convert karna easy mapping ke liye
            scores = {res['label'].lower(): res['score'] for res in pipeline_output}
            
            # Model ke output labels ko standardize karna (kuch models mein 'positive' hota hai, kuch mein 'label_2')
            # CardiffNLP RoBERTa labels: 'negative', 'neutral', 'positive'
            
            # Best label nikalna based on highest score
            best_label = max(scores, key=scores.get)
            confidence_score = scores[best_label]
            
            # Next-Level Logic: Agar top score bohot kam hai, toh automatically 'neutral' mark karein
            if confidence_score < 0.45:
                return "neutral", confidence_score
                
            return best_label, confidence_score
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "neutral", 0.50

# --- Test Framework ---
if __name__ == "__main__":
    engine = NextGenHinglishEngine()
    
    test_sentences = [
        "Product bohot sahi hai, maza aa gaya!", # Positive Hinglish
        "Bilkul bekar service hai, delivery bahut delay hui.", # Negative Hinglish
        "Thik thak hai, na zyada acha na bura.", # Neutral Hinglish
        "Maza nahi aya bhai." # Complex Negative (Keyword 'maza' hai par sentiment negative hai)
    ]
    
    print("\n--- Testing Results ---")
    for text in test_sentences:
        label, score = engine.predict_sentiment(text)
        print(f"Text: '{text}' -> Sentiment: {label.upper()} ({score:.2%})")