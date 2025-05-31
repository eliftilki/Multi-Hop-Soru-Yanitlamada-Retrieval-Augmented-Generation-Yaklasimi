
import os
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import sacrebleu
import psutil
import huggingface_hub


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HotpotRAGSystem:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=None):
        """Initialize the RAG system."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.questions = []
        self.answers = []
        self.contexts = []
        

        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension) 
        
        # FLAN-T5 model ayarları
        try:
            print("FLAN-T5-base modeli yükleniyor...")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base",
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("FLAN-T5-base modeli başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
            self.generator = None
            self.tokenizer = None
        
        # Metrik hesaplayıcılar
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)  # METEOR için gerekli
    
    def load_hotpot_data(self, data_path: str, max_samples: int = None):
        """HotpotQA JSON formatındaki verileri yükler ve işler."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Tüm veri setini kullan
            print(f"Toplam {len(data)} veri örneği yüklendi.")
            
            processed_data = []
            for idx, item in enumerate(data):
                try:
                    # HotpotQA formatından bizim formatımıza dönüştür
                    context = []
                    for paragraph in item['context']:
                        title = paragraph[0]
                        content = paragraph[1]
                        context.append({
                            'baslik': title,
                            'icerik': [content]
                        })
                    
                    processed_data.append({
                        'question': item['question'],
                        'context': context,
                        'answer': item['answer'],
                        'question_length': len(item['question'].split()),
                        'answer_length': len(item['answer'].split()),
                        'context_paragraph_count': len(context)
                    })
                    
                except Exception as e:
                    print(f"Hata: {idx}. öğe işlenirken hata oluştu: {str(e)}")
                    continue
            
            print(f"Başarıyla işlenen veri sayısı: {len(processed_data)}")
            return processed_data
            
        except Exception as e:
            print(f"Dosya okuma hatası: {str(e)}")
            raise

    def process_context(self, context_list: List[Dict]) -> List[str]:
        """Context listesini düz metin formatına dönüştürür."""
        processed_context = []
        for paragraph in context_list:
            title = paragraph['baslik']
            for content in paragraph['icerik']:
                processed_context.append(f"{title}: {content}")
        return processed_context

    def find_relevant_context(self, question: str, context_list: List[Dict], k: int = 3) -> List[str]:
        """Soru için en alakalı context paragraflarını bulur."""
        processed_context = self.process_context(context_list)
        return self.retrieve(question, k)

    def answer_question(self, question: str, context_list: List[Dict]) -> str:
        """Soruya cevap üretir."""
        # Context'i metin listesine dönüştür
        context_texts = [f"{item['baslik']}: {item['icerik'][0] if item['icerik'] else ''}" for item in context_list]
        return self.generate_answer(question, context_texts)
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Metinler için embedding'ler oluşturur."""
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Embedding oluşturuluyor"):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    normalize_embeddings=True  # Embedding'leri normalize et
                )
                all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)
    
    def build_index(self, data: List[Dict], batch_size: int = 32):
        """FAISS indeksini oluşturur ve kaydeder."""
        self.documents = []
        self.questions = []
        self.answers = []
        self.contexts = []
        print("Veri hazırlanıyor...")
        for item in data:
            processed_contexts = []
            for paragraph in item['context']:
                title = paragraph['baslik']
                for content in paragraph['icerik']:
                    processed_contexts.append(f"{title}: {content}")
            self.documents.extend(processed_contexts)
            self.questions.append(item['question'])
            self.answers.append(item['answer'])
            self.contexts.append(item['context'])
        print(f"Toplam {len(self.documents)} döküman işlenecek.")
        self.index = faiss.IndexFlatIP(self.dimension)
        # Batch'ler halinde embedding oluşturma ve indekse ekleme
        total_batches = (len(self.documents) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(self.documents), batch_size), total=total_batches, desc="İndeks oluşturuluyor"):
            batch_docs = self.documents[i:i + batch_size]
            batch_embeddings = self.create_embeddings(batch_docs, batch_size=batch_size)
            self.index.add(batch_embeddings.astype('float32'))
        # İndeksi kaydetme
        print("İndeks kaydediliyor...")
        with open("embeddings.pkl", 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'documents': self.documents,
                'questions': self.questions,
                'answers': self.answers,
                'contexts': self.contexts
            }, f)
        print("İndeks oluşturma tamamlandı.")
    
    def load_index(self):
        """Kaydedilmiş indeksi yükler."""
        if os.path.exists("embeddings.pkl"):
            print("Kaydedilmiş indeks yükleniyor...")
            with open("embeddings.pkl", 'rb') as f:
                saved_data = pickle.load(f)
            
            # İndeksi oluşturma
            self.dimension = saved_data['dimension']
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Verileri yükleme
            self.documents = saved_data['documents']
            self.questions = saved_data['questions']
            self.answers = saved_data['answers']
            self.contexts = saved_data['contexts']
            
            # Batch'ler halinde embedding'leri yükleme ve indekse ekleme
            batch_size = 32
            total_batches = (len(self.documents) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(self.documents), batch_size), total=total_batches, desc="İndeks yükleniyor"):
                batch_docs = self.documents[i:i + batch_size]
                batch_embeddings = self.create_embeddings(batch_docs, batch_size=batch_size)
                self.index.add(batch_embeddings.astype('float32'))
            
            print("İndeks yükleme tamamlandı.")
            return True
        return False
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Sorgu için en alakalı k metni getirir."""
        if not self.documents:
            print("Uyarı: Henüz döküman yüklenmemiş.")
            return []
        if not self.index or self.index.ntotal == 0:
            print("Uyarı: İndeks boş veya oluşturulmamış.")
            return []
        k = min(k, len(self.documents))
        try:
            with torch.no_grad():
                query_embedding = self.model.encode(
                    [query],
                    convert_to_numpy=True,
                    batch_size=1,
                    normalize_embeddings=True  # Sorgu embedding'ini de normalize et
                )
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            if len(indices) == 0 or len(indices[0]) == 0:
                print("Uyarı: İndeks araması sonuç döndürmedi.")
                return []
            return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]
        except Exception as e:
            print(f"İndeks arama hatası: {str(e)}")
            return []
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generates an answer using context and question."""
        if not self.generator or not self.tokenizer:
            return "Üzgünüm, model yüklenemediği için cevap üretemiyorum."
        
        context_text = ' '.join(context)
        input_text = f"question: {question} context: {context_text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def evaluate_retrieval(self, retrieved_texts: List[str], supporting_facts: List[Tuple[str, str]]) -> float:
        """Retrieval performansını değerlendirir."""
        supporting_texts = [f"{title}: {text}" for title, text in supporting_facts]
        correct = sum(1 for text in retrieved_texts if text in supporting_texts)
        return correct / len(retrieved_texts) if retrieved_texts else 0
    
    def calculate_metrics(self, predicted: str, reference: str) -> Dict[str, float]:
        """Çeşitli metrikleri hesaplar."""
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
    
    def evaluate_system(self, max_samples: int = None):
        """Evaluates the entire system."""
        print("\nEvaluating system...")
        
        print("Creating index...")
        embeddings = self.model.encode(self.documents)
        self.index.add(embeddings.astype('float32'))
        
        questions = self.questions
        answers = self.answers
        
        total_questions = len(questions)
        correct_answers = 0
        
        # Metrikler için toplam değerler
        total_metrics = {
            'bleu': 0.0,
            'meteor': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'f1': 0.0
        }
        
        print(f"\nToplam {total_questions} soru üzerinde değerlendirme yapılıyor...")
        
        # Her 100 soruda bir ilerleme gösterir
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            if i % 100 == 0:
                print(f"\nİşlenen soru: {i}/{total_questions}")
                print(f"Şu ana kadar doğru cevap sayısı: {correct_answers}")
                print(f"Şu ana kadar doğruluk: {(correct_answers/i)*100:.2f}%")
                print("\nAra Metrikler:")
                for metric_name, value in total_metrics.items():
                    print(f"{metric_name.upper()}: {value/i:.4f}")
            
           # Context getirme ve cevabı oluşturma
            context = self.retrieve(question)
            generated_answer = self.generate_answer(question, context)
            
            # Eşleşme Hesaplama
            if not generated_answer:
                generated_answer = ""
            if generated_answer.lower().strip() == answer.lower().strip():
                correct_answers += 1
            
            # Metrikleri hesapla
            metrics = self.calculate_metrics(generated_answer, answer)
            for metric_name, value in metrics.items():
                total_metrics[metric_name] += value
        
        # Final sonuçları
        accuracy = (correct_answers / total_questions) * 100
        print(f"\nDeğerlendirme tamamlandı!")
        print(f"Toplam soru sayısı: {total_questions}")
        print(f"Doğru cevap sayısı: {correct_answers}")
        print(f"Doğruluk: {accuracy:.2f}%")
        
        # Final metrikleri
        print("\nFinal Metrikler:")
        for metric_name, value in total_metrics.items():
            print(f"{metric_name.upper()}: {value/total_questions:.4f}")

# Ön işlem fonksiyonu ekle (dosyanın ilk num_lines satırını alıp JSON array haline getirir ve ilk satırı ekrana basar)
def preprocess_json(input_path, output_path, num_lines=2):
    """Ön işlem: Dosyanın ilk num_lines satırını alıp, JSON array haline getirir."""
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()[:num_lines]
        # İlk satırı ekrana basma
        # Eğer dosya zaten bir array ise, bu adımı atla
        try:
            data = json.loads("".join(lines))
            if isinstance(data, list):
                return input_path
        except Exception:
            # Satır satır JSON objesi ise, ilk num_lines satırı alıp array haline getir
            data = [json.loads(line) for line in lines if line.strip()]
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)
    print("Ön işlem tamamlandı. Yeni dosya kaydedildi:", output_path)
    return output_path


def main():
    # Veri dosyası yolu
    data_path = r"C:\Users\Sude\Desktop\LLMProje\data\hotpot_dev_distractor_v1.json"
    
    
    rag = HotpotRAGSystem()
    
    # Veriyi yükle (tüm veri seti)
    print("Veri yükleniyor (tüm veri seti)...")
    data = rag.load_hotpot_data(data_path) 
    print(f"Yüklenen veri sayısı: {len(data)}")
    
    # İndeks oluştur veya yükle
    if not rag.load_index():
        print("İndeks oluşturuluyor...")
        rag.build_index(data, batch_size=8)
    
    # Sistem değerlendirmesi (tüm veri seti üzerinde)
    print("\nSistem değerlendiriliyor (tüm veri seti üzerinde)...")
    rag.evaluate_system()  

if __name__ == "__main__":
    main() 