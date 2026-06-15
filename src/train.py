import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import HinglishPreprocessor

def start_training():
    path = "data/real_hinglish_dataset.csv"
    if not os.path.exists(path):
        print(f"❌ Error: {path} nahi mila! Pehle download_data.py run karo.")
        return

    print("📦 Step 1: Loading Real Massive Hinglish Dataset...")
    df = pd.read_csv(path)
    
    df = df.dropna(subset=['text', 'label'])
    
    print(f"🧹 Step 2: Running Hinglish Preprocessing Engine on {len(df)} rows...")
    preprocessor = HinglishPreprocessor()
    df['cleaned_text'] = df['text'].apply(preprocessor.deep_clean)
    
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print("🔤 Step 3: Extracting Features using TF-IDF (Char N-grams)...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=25000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("🤖 Step 4: Training Baseline Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    print("📊 Step 5: Evaluating Model Performance on Real Data...")
    predictions = model.predict(X_test_tfidf)
    
    print("\n================ REAL DATA BASELINE RESULTS ================")
    print(f"Accuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("============================================================")
    
    print("💾 Saving baseline model components for production...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/baseline_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("✅ Components saved successfully inside 'models/' folder!")

if __name__ == "__main__":
    start_training()