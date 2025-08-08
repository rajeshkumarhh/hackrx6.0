# 📦 Import required packages

from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 📄 Set your local PDF path here
pdf_path = "C:/Users/rajeshkumar/Desktop/upload/doc1.pdf"  # <-- Change this to your actual file path

# 📚 PDF Text Extraction
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# 🧩 Split text into chunks
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ⚡ Load Fast Summarization Model
model_id = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Check if CUDA (GPU) is available
device = 0 if torch.cuda.is_available() else -1
if device == -1:
    print("⚠️ CUDA (GPU) not available. Using CPU, which may be slower.")

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

# 🧠 Summarize each chunk
def summarize_chunks(chunks):
    summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=True)[0]['summary_text']
        print(f"\n📌 Summary {i+1}:\n{summary}\n")
        summaries.append(summary)
    return summaries

# 🚀 Run the pipeline
print("✅ Found PDF:", pdf_path)
policy_text = extract_pdf_text(pdf_path)
chunks = chunk_text(policy_text)
summaries = summarize_chunks(chunks)