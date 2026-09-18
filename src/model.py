from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def create_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = create_pipeline()
    print("🚀 Training classification pipeline...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model
