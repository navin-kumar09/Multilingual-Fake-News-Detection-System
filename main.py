from sklearn.model_selection import train_test_split
from src.data_loader import build_combined_dataset
from src.translation import detect_language, translate_hi_to_en
from src.model import train_and_evaluate

def main():
    # 1. Load and build dataset
    df = build_combined_dataset()
    if df.empty:
        return
        
    X = df["text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Train model
    model = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 3. Test Multilingual Prediction
    sample_text = "मुझे मुफ्त पेट्रोल मिलेगा"
    lang = detect_language(sample_text)
    print(f"\n[Test] Input: '{sample_text}' | Detected Language: {lang}")
    
    eval_text = translate_hi_to_en(sample_text) if lang == "hi" else sample_text
    pred = model.predict([eval_text])[0]
    
    result = "Real News 🟢" if pred == 1 else "Fake News 🔴"
    print(f"[Test] Prediction Result: {result}")

if __name__ == "__main__":
    main()
