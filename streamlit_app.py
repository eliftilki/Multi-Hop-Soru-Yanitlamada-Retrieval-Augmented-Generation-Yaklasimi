import streamlit as st
import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Başlık
st.set_page_config(page_title="RAG QA Sistemi", layout="centered")
st.title("💬 Multi-Hop HotpotQA Soru Yanıtlama Sistemi")

# Verileri yükle
@st.cache_resource
def load_all_resources():
    with open("hotpot_dev_distractor_v1.json", "r") as f:
        data = json.load(f)
    paragraph_map = json.load(open("paragraph_map.json"))
    paragraphs_all = [ctx for ex in data for _, ctx_list in ex["context"] for ctx in [" ".join(ctx_list)]]

    retriever = SentenceTransformer("intfloat/e5-large-v2")
    index = faiss.read_index("faiss_index.index")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to("cuda" if torch.cuda.is_available() else "cpu")

    return data, paragraph_map, paragraphs_all, retriever, index, tokenizer, model

data, paragraph_map, paragraphs_all, retriever, index, tokenizer, model = load_all_resources()

# Fonksiyon: Soruya cevap üret
def answer_question_with_rag(question, top_k=3):
    q_embedding = retriever.encode(question, convert_to_numpy=True)
    _, I = index.search(np.array([q_embedding]), top_k)
    selected_paragraphs = [paragraphs_all[i] for i in I[0]]
    input_text = f"question: {question} context: {' '.join(selected_paragraphs)}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True), selected_paragraphs

# Kullanıcıdan giriş al
question = st.text_input("🔎 Sorunuzu aşağıya yazın, ben de sizin için cevabını bulayım!")

if st.button("Cevapla") and question.strip():
    with st.spinner("Cevap üretiliyor..."):
        pred, paras = answer_question_with_rag(question)

        st.subheader("📌 Cevap")
        st.success(pred)

        # Altın cevap varsa göster
        gold_answer = None
        for ex in data:
            if ex["question"].lower().strip() == question.lower().strip():
                gold_answer = ex["answer"]
                break

        if gold_answer:
            st.markdown(f"**🟡 Altın Cevap:** {gold_answer}")

        st.subheader("📚 En İlgili Paragraflar")
        for i, p in enumerate(paras, 1):
            st.markdown(f"**Paragraf {i}:** {p}")
