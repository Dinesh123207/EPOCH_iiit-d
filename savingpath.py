# cell1: Language Detection
import langid

def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang

# Example usage
sentence = "تعلیم اور سائنس معاشرتی ترقی اور فردی ترقی میں اہم کردار ادا کرتے ہیں۔ یہ دونوں ہمیں نے ایک نیا دیدار دینے میں مدد کی ہے اور زندگی کو بہتر بنانے میں مدد کی ہے۔تعلیم ہر انسان کی شخصیت کو بناتی ہے اور اسے معاشرتی چیلنجز کا مقابلہ کرنے میں مدد فراہم کرتی ہے۔ اچھی تعلیم ایک ملک کو پیشہ ورانہ اور ترقی یافتہ بناتی ہے اور لوگوں کو مختلف حلقوں میں فراہم کرتی ہے۔سائنس نے ہمیں تکنالوجی کے ذریعے زندگی کو آسان بنایا ہے۔ یہ ہمیں نئے میڈیکل ترکیبوں، بھتہی اور مواد کا استعمال کرنے کا موقع دیتا ہے جو صحت اور رفاہت میں بہتری لاتا ہے۔تعلیم اور سائنس کا مشترک اہم کردار ہے جو ہمیں بہترین ممکنہ مستقبل کی طرف لے جا رہا ہے۔ یہ ہمیں سمجھ, تربیت اور پیشہ ورانہ تربیت فراہم کرتا ہے جو زندگی میں کامیابی حاصل کرنے میں مدد فراہم کرتا ہے۔اگر ہم تعلیم اور سائنس کی قدر کریں تو ہم زندگی کو محنتی اور خوبصورت بنا سکتے ہیں اور اپنے معاشرتی ہتھیاروں کو بھی مزید بڑھا سکتے ہیں۔"
lang = detect_language(sentence)
if lang=="ne":
  source_lang="hi"
elif lang=="fa":
  source_lang="ur"
else:
  source_lang=lang
  
print(f"The detected language is: {source_lang}")

# cell2: Sentence Splitting
if source_lang != "en":
    import re

    def split_into_sentences(text):
        sentences = re.split(r'[।.!?]', text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences

    input_text = sentence
    sentences = split_into_sentences(input_text)

    print("\nList of Sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence}")

# cell3: English Check
if source_lang == "en":
    combined_translation = sentence
# cell4: Model Saving
import os
from transformers import MarianMTModel, MarianTokenizer

# Check if models are already saved by reading from a file
models_saved_filepath = "models_saved.txt"
models_saved = os.path.exists(models_saved_filepath)

global models
global tokenizers
models = {}
tokenizers = {}

# If models are already saved, load models and tokenizers from the saved path
language_pairs = [
        {"source": "pa", "target": "en"},
        {"source": "hi", "target": "en"},
        {"source": "ur", "target": "en"},
        {"source": "bn", "target": "en"},
        {"source": "en", "target": "en"},
    ]
if models_saved:
    print("Models are already saved. Loading models and tokenizers.")
    for pair in language_pairs:
        source_lang1 = pair["source"]
        target_lang1 = pair["target"]

        if source_lang1 == target_lang1:
            continue

        model_name = f"Helsinki-NLP/opus-mt-{source_lang1}-{target_lang1}"
        saved_path = f"{model_name}_pretrained_model_pytorch"

        model = MarianMTModel.from_pretrained(saved_path, return_dict=True)
        tokenizer = MarianTokenizer.from_pretrained(saved_path)

        models[(source_lang1, target_lang1)] = model
        tokenizers[(source_lang1, target_lang1)] = tokenizer
else:
    def save_translation_models(language_pairs):
        global models_saved

        # Check if models are already saved
        if models_saved:
            print("Models are already saved. Skipping model saving.")
            return

        saved_model_paths = []

        for pair in language_pairs:
            source_lang1 = pair["source"]
            target_lang1 = pair["target"]

            if source_lang1 == target_lang1:
                continue

            model_name = f"Helsinki-NLP/opus-mt-{source_lang1}-{target_lang1}"
            save_path = f"{model_name}_pretrained_model_pytorch"

            model = MarianMTModel.from_pretrained(model_name, return_dict=True)
            tokenizer = MarianTokenizer.from_pretrained(model_name)

            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            models[(source_lang1, target_lang1)] = model
            tokenizers[(source_lang1, target_lang1)] = tokenizer

            saved_model_paths.append(save_path)

        for path in saved_model_paths:
            print(f"Model saved at: {path}")

        # Set the flag to True after saving the models
        models_saved = True

        # Save the models_saved flag to a file
        with open(models_saved_filepath, "w") as file:
            file.write("1")

    # Example usage for saving models

    save_translation_models(language_pairs)

# cell5: Translation
combined_translation = ""
if source_lang != "en":
    def translate_sentences(sentences, source_lang, target_lang):
        if (source_lang, target_lang) in models:
            model = models[(source_lang, target_lang)]
            tokenizer = tokenizers[(source_lang, target_lang)]

            translated_texts = []

            for sentence in sentences:
                inputs = tokenizer(sentence, return_tensors="pt")
                outputs = model.generate(**inputs)
                translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translated_texts.append(translated_text[0])

            return translated_texts
        else:
            return [f"Model for {source_lang}-{target_lang} not available."] * len(sentences)
    target_lang = "en"
    translated_texts = translate_sentences(sentences, source_lang, target_lang)
    combined_translation = ".".join(translated_texts)

# cell6: Print Combined Translation
print("Combined Translation:")
print(combined_translation)


