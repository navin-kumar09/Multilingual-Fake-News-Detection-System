from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DetectorFactory.seed = 0

print("🔄 Loading Helsinki-NLP translation models...")
hi_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
hi_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-hi-en")

def detect_language(text):
    try:
        lang = detect(text)
        if lang == "mr": 
            return "hi"  # Marathi/Hindi heuristic fix
        return lang
    except Exception:
        return "unknown"

def translate_hi_to_en(text):
    try:
        inputs = hi_en_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = hi_en_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        return hi_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        return text
