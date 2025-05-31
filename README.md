# 💬 Multi-Hop HotpotQA Soru Yanıtlama Sistemi

Bu proje, [HotpotQA](https://hotpotqa.github.io/) veri seti üzerinde çalışan *RAG (Retrieval-Augmented Generation)* tabanlı bir soru yanıtlama sistemidir. Kullanıcıdan alınan soruya uygun olarak en ilgili paragrafları bulur ve ardından T5 tabanlı bir jeneratif modelle cevap üretir.

## 🚀 Özellikler

    🔍 *Soru-Cevap*: Kullanıcının girdiği soruya en uygun yanıtı üretir.
    📚 *Paragraf Getirme*: FAISS ve SentenceTransformer kullanarak ilgili bağlamları bulur.
    🧠 *Model*: google/flan-t5-large jeneratif dil modeli ile cevap üretir.
    ⚡ *Hızlı*: FAISS ile hızlı vektör arama, GPU desteği (varsa) ile hızlı cevaplama.
    🎨 *Arayüz*: Streamlit tabanlı kullanıcı dostu görsel arayüz.


## 🖼️ Uygulama Görselleri

### Ana Sayfa (Soru sorulmadan önce)
![Uygulama Görseli 1](images/main_screen.png)

### Cevaplama Ekranı (Soru sorulduktan sonra)
![Uygulama Görseli 2](images/answer_screen.png)

## 🧩 Kullanılan Teknolojiler

| Bileşen             | Açıklama                                       |
|---------------------|------------------------------------------------|
| Streamlit           | Web tabanlı arayüz                             |
| SentenceTransformers| Soru ve metin gömme işlemi                     |
| FAISS               | Paragraf vektörleri üzerinde hızlı arama       |
| Transformers        | T5 tabanlı jeneratif model (flan-t5-large)  |
| HotpotQA            | Çok adımlı mantıksal çıkarım gerektiren veri seti |

## 🛠️ Kurulum

```bash
git clone https://github.com/kullaniciadi/hotpotqa-rag-streamlit.git
cd hotpotqa-rag-streamlit

# Gerekli paketleri yükle
pip install -r requirements.txt