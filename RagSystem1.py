import json
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Tuple
from tqdm import tqdm
import random
from difflib import SequenceMatcher
from collections import Counter
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from textdistance import jaro_winkler
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str, use_lemmatizer: bool = False) -> str:
   """
    Verilen metni küçük harfe çevirir, noktalama işaretlerini ve stopword'leri temizler.
    İsteğe bağlı olarak lemmatization uygular.
    """
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(token) if use_lemmatizer else token
        for token in tokens if token not in stop_words
    ]
    return ' '.join(filtered_tokens).strip()

class RAGSystem:
  """
    RAG (Retrieval-Augmented Generation) tabanlı Soru-Cevaplama sistemi.
    """
    def __init__(self,
                 retriever_model_name: str = "intfloat/e5-large-v2",
                 generator_model_name: str = "google/flan-t5-large",
                 reranker_model_name: str = "cross-encoder/stsb-roberta-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Kullanılan cihaz: {self.device}")

        self.retriever = SentenceTransformer(retriever_model_name).to(self.device)
        self.reranker = CrossEncoder(reranker_model_name).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name).to(self.device)

        self.index = None
        self.documents = []
        self.doc_embeddings = None
        self.paragraph_map = {}

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
      """
        Uzun metinleri belirli kelime uzunluğuna göre parçalara böler.
        """
        tokens = word_tokenize(text)
        return [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    def load_documents(self, json_file_path: str, chunk_size: int = 200):
      """
        JSON dosyasından belgeleri yükler, ön işler ve FAISS index oluşturur.
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents = []
        self.paragraph_map = {}
        doc_id = 0

        for item in data:
            example_id = item.get("id", str(random.randint(0, 999999)))
            context = item.get('context', [])
            for para_idx, (title, paragraphs) in enumerate(context):
                for paragraph in paragraphs:
                    chunks = self.chunk_text(paragraph, chunk_size=chunk_size)
                    for chunk in chunks:
                        processed = preprocess_text(chunk)
                        if processed:
                            self.documents.append(processed)
                            self.paragraph_map[doc_id] = (example_id, para_idx)
                            doc_id += 1

        print("Embedding'ler hesaplanıyor...")
        self.doc_embeddings = self.retriever.encode(self.documents, show_progress_bar=True, batch_size=32)
        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.add(self.doc_embeddings.astype('float32'))

    def save_index(self, path="rag_index"):
      """
        FAISS index ve dokümanları belirtilen dizine kaydeder.
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        joblib.dump((self.documents, self.paragraph_map), os.path.join(path, "documents.pkl"))
        print("Index ve dokümanlar kaydedildi.")

    def load_index(self, path="rag_index"):
      """
        Daha önce kaydedilen FAISS index ve dokümanları yükler.
        """
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        self.documents, self.paragraph_map = joblib.load(os.path.join(path, "documents.pkl"))
        print("Index ve dokümanlar yüklendi.")

    def retrieve(self, query: str, k: int = 10) -> Tuple[List[str], List[int]]:
       """
        Sorgu için en benzer k dokümanı getirir.
        """
        query_embedding = self.retriever.encode([preprocess_text(query)], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.documents[i] for i in indices[0]], indices[0].tolist()

    def rerank(self, query: str, docs: List[str], indices: List[int], top_k: int = 5) -> Tuple[List[str], List[int]]:
       """
        Alınan dokümanları reranker modeliyle yeniden sıralar.
        """
        pairs = [[query, doc] for doc in docs]
        scores = self.reranker.predict(pairs, convert_to_numpy=True, batch_size=16)
        ranked = sorted(zip(docs, indices, scores), key=lambda x: x[2], reverse=True)
        top_docs = [doc for doc, _, _ in ranked[:top_k]]
        top_indices = [idx for _, idx, _ in ranked[:top_k]]
        return top_docs, top_indices

    def clean_generated_text(self, text: str) -> str:
      """
        Modelin ürettiği cevaptaki gereksiz boşlukları ve noktaları temizler.
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.*$', '', text)
        return text.strip()

    def generate_answer(self, query: str, context: List[str]) -> str:
      """
        Verilen sorgu ve bağlam kullanılarak modelden cevap üretilir.
        """
        context_text = " ".join(context)
        prompt = (
            f"Use the following context to answer the question. "
            f"The answer should be concise and based *only* on the provided information.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.generator.generate(
                inputs["input_ids"],
                max_length=150,
                num_beams=5,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )
        return self.clean_generated_text(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    def answer_question(self, query: str, k: int = 10, rerank_k: int = 5) -> Tuple[str, List[int]]:
      """
        Tüm pipeline: sorguya cevap üretir ve kullanılan doküman indekslerini döner.
        """
        retrieved_docs, indices = self.retrieve(query, k)
        reranked_docs, top_indices = self.rerank(query, retrieved_docs, indices, top_k=rerank_k)
        return self.generate_answer(query, reranked_docs), top_indices

    def calculate_metrics(self, predicted: str, reference: str) -> Dict[str, float]:
         """
        Üretilen cevap ile altın standart arasında metrikleri hesaplar (BLEU, METEOR, ROUGE, F1).
        """
        metrics = {}

        # BLEU
        reference_tokens = nltk.word_tokenize(reference.lower())
        predicted_tokens = nltk.word_tokenize(predicted.lower())
        metrics['bleu'] = sentence_bleu([reference_tokens], predicted_tokens)

        # ROUGE
        rouge_scores = self.rouge_scorer.score(reference, predicted)
        metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
        metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
        metrics['rougeL'] = rouge_scores['rougeL'].fmeasure

        # METEOR
        try:
            metrics['meteor'] = meteor_score([reference_tokens], predicted_tokens)
        except Exception as e:
            print(f"METEOR hesaplama hatası: {str(e)}")
            metrics['meteor'] = 0.0

        # F1 Score (token bazlı)
        reference_set = set(reference_tokens)
        predicted_set = set(predicted_tokens)
        common = reference_set.intersection(predicted_set)
        precision = len(common) / len(predicted_set) if predicted_set else 0
        recall = len(common) / len(reference_set) if reference_set else 0
        metrics['f1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return metrics

    def evaluate_retrieval(self, retrieved_texts: List[str], supporting_facts: List[Tuple[str, str]]) -> float:

        """
        Destekleyici gerçeklerle eşleşen retrieval başarımını hesaplar.
        """
        supporting_texts = [f"{title}: {text}" for title, text in supporting_facts]
        correct = sum(1 for text in retrieved_texts if text in supporting_texts)
        return correct / len(retrieved_texts) if retrieved_texts else 0

    def evaluate(self, test_file: str, k: int = 10, rerank_k: int = 5) -> Dict[str, float]:
      """
        Test veri seti ile sistemin genel performansını değerlendirir ve metrikleri dosyaya yazar.
        """
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # Metrikler için toplam değerler
        total_metrics = {
            'bleu': 0.0,
            'meteor': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'f1': 0.0,
            'retrieval': 0.0 
        }

        total_questions = len(test_data)
        correct_answers = 0

        print(f"\nToplam {total_questions} soru üzerinde değerlendirme yapılıyor...")

        for i, example in enumerate(tqdm(test_data, desc="Değerlendiriliyor"), 1):
            reference = example['answer']
            prediction, top_k_indices = self.answer_question(example['question'], k, rerank_k)

            # Simple exact match evaluation
            if prediction.lower().strip() == reference.lower().strip():
                correct_answers += 1

            # Metrikleri hesapla
            metrics = self.calculate_metrics(prediction, reference)
            for metric_name, value in metrics.items():
                total_metrics[metric_name] += value

            # Retrieval değerlendirmesi
            retrieved_docs = [self.documents[idx] for idx in top_k_indices]
            retrieval_score = self.evaluate_retrieval(retrieved_docs, example["supporting_facts"])
            total_metrics['retrieval'] += retrieval_score

            # Her 100 soruda bir ilerleme göster
            if i % 100 == 0:
                print(f"\nİşlenen soru: {i}/{total_questions}")
                print(f"Şu ana kadar doğru cevap sayısı: {correct_answers}")
                print(f"Şu ana kadar doğruluk: {(correct_answers/i)*100:.2f}%")
                print("\nAra Metrikler:")
                for metric_name, value in total_metrics.items():
                    print(f"{metric_name.upper()}: {value/i:.4f}")

        # Final sonuçları
        accuracy = (correct_answers / total_questions) * 100
        final_metrics = {
            "ACCURACY": accuracy,
            "BLEU": total_metrics['bleu'] / total_questions,
            "METEOR": total_metrics['meteor'] / total_questions,
            "ROUGE1": total_metrics['rouge1'] / total_questions,
            "ROUGE2": total_metrics['rouge2'] / total_questions,
            "ROUGEL": total_metrics['rougeL'] / total_questions,
            "F1": total_metrics['f1'] / total_questions,
            "RETRIEVAL": total_metrics['retrieval'] / total_questions  # Retrieval metriği eklendi
        }

        print(f"\nDeğerlendirme tamamlandı!")
        print(f"Toplam soru sayısı: {total_questions}")
        print(f"Doğru cevap sayısı: {correct_answers}")
        print(f"Doğruluk: {accuracy:.2f}%")

        print("\nFinal Metrikler:")
        for metric_name, value in final_metrics.items():
            print(f"{metric_name}: {value:.4f}")

        with open("/content/drive/MyDrive/rag_eval_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        return final_metrics

def main():
  """
    Sistemi başlatır, index’i yükler veya oluşturur ve kullanıcıdan gelen soruları cevaplar.
    """
    rag = RAGSystem()
    index_path = "/content/drive/MyDrive/rag_index"

    if os.path.exists(os.path.join(index_path, "faiss.index")):
        print("Kaydedilmiş index yüklenecek...")
        rag.load_index(index_path)
    else:
        print("Index oluşturuluyor ve kaydediliyor...")
        rag.load_documents("/content/drive/MyDrive/hotpot_dev_distractor_v1.json", chunk_size=200)
        rag.save_index(index_path)

    print("\nTest veri seti ile değerlendirme başlıyor...")
    rag.evaluate("/content/drive/MyDrive/hotpot_dev_distractor_v1.json")

    while True:
        query = input("\n❓ Sormak istediğiniz bir soru girin (çıkmak için 'exit'): ")
        if query.lower() == "exit":
            break
        answer, _ = rag.answer_question(query)
        print(f"💬 Cevap: {answer}")

if __name__ == "__main__":
    main()