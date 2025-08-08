HACKRX6.0
---

## 🛠️ Project Overview: GramFinance + DeFi

A decentralized finance platform designed to empower rural communities by offering micro-loans, savings, and insurance through local agents. Key features include:
- **Ethereum smart contracts** for secure financial operations
- **WhatsApp-based AI assistant** for multilingual, voice-enabled services
- Built for the **Agentic Ethereum Hackathon – India**

---

## 🤖 Autonomous DeFi Agent

- Uses **LlamaIndex + GPT-powered chatbots** to onboard users, manage loans, and generate financial insights
- Received positive feedback from industry mentors for its innovative approach to financial inclusion

---

## 📄 Document Summarization System

### Workflow:
1. **Ingest & Preprocess**: Accepts `.pdf`, `.csv`, `.docx` files; extracts text (PyPDF) and tables (Camelot)
2. **LLM Summarization**: Generates separate summaries for text and tables using GPT
3. **Merge & Finalize**: Combines both into a concise, actionable summary

---

## 🧰 Tech Stack

| Layer       | Tools Used                                                                 |
|-------------|-----------------------------------------------------------------------------|
| Cloud       | AWS (S3, EC2) or Azure                                                      |
| Database    | PostgreSQL (metadata), MongoDB (summaries)                                 |
| Backend     | Python (FastAPI), LlamaIndex, OpenAI GPT API                               |
| Frontend    | Streamlit (demo UI) or React.js (production dashboard)                     |

---

## 🧬 Architecture Highlights

- Dual pipeline for **text and table processing**
- GPT-based summarization for **policy paragraphs and compliance tables**
- Outputs in **executive overview**, **bullet points**, and **structured JSON**
- Scalable for cloud deployment and adaptable across domains (legal, HR, finance)

---

## 🌟 Unique Selling Points

- **Dual-pipeline** for complete detail capture
- **Custom GPT prompts** for accurate extraction
- **Structured outputs** for easy integration
- **Domain adaptability** and **scalability**

---

## 🚀 Future Enhancements

- Multilingual summarization
- Auto-updating policies
- Smart clause search via natural language
- Integration with enterprise systems
- Visual insights and self-learning models

---

## ⚠️ Risks & Challenges

- Managing sensitive data securely
- Ensuring contextual accuracy in legal/compliance terms
- Handling diverse document formats
- Maintaining performance under load
- Dependence on third-party APIs
- Adapting to evolving regulations

---

## ✅ Acceptance Criteria Coverage

The solution meets the hackathon’s requirements by:
- Handling both text and tabular data
- Extracting actionable insights
- Supporting multiple formats and scalable deployment
- Providing structured outputs for enterprise integration
- Ensuring adaptability, accuracy, and security

---
