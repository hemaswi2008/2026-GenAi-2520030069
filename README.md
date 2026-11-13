Prompting vs RAG vs Fine-Tuning for Customer Support Automation

A project that compares three ways of adapting a Large Language Model (LLM) for customer support automation: Prompting, Retrieval-Augmented Generation (RAG) and Fine-Tuning. The comparison is done on support queries written in English, Telugu and Hindi, and covers answer quality, reliability, tokenization and cost.

Table of Contents
Introduction
Problem Statement
Objectives
The Three Approaches
Tokenization and Multilingual Support
Token Economics and Cost Estimation
Methodology
Evaluation Metrics
Tech Stack
Project Structure
Setup and Usage
Expected Outcome
Introduction

Generative AI has quickly become part of everyday business tools, and customer support is one of its most common uses. LLMs are foundation models trained on very large text collections using next-token prediction. They can draft replies, summarise tickets and answer product questions in natural language.

Using an LLM in a real support setting is not simple, though. A general model has limits that matter a lot for support:

Hallucinations: the model can give confident but wrong answers about refunds, prices or policies.
Knowledge cutoff: the model does not know about products, offers or policy changes after its training date.
Pricing, rate limits and failure rates: hosted models charge per token, limit how many requests can be made, and sometimes fail, which affects how many tickets can be handled reliably.
Open-weight vs closed models: closed models are easy to use but give less control over data and deployment, while open-weight models can be hosted and tuned but need more setup.
Problem Statement

Companies receive large numbers of customer queries about orders, refunds, billing and product use, and answering them by hand is slow and costly. A general LLM does not know a company's own policies and may give misleading answers.

There are three common ways to adapt an LLM to a business: Prompting, RAG and Fine-Tuning. Each differs in accuracy, cost, effort and maintenance, and there is little clear guidance on which to choose. The problem is harder in India, where customers write in English, Telugu and Hindi, and the tokenization of these languages changes token counts and cost.

Research question: Which approach gives the most accurate, reliable and affordable customer support automation across these languages?

Objectives
Understand how LLMs work (next-token prediction) and where they fail in customer support.
Study hallucinations, knowledge cutoffs, pricing, rate limits and failure rates.
Implement Prompting, RAG and Fine-Tuning on the same set of support queries.
Compare tokenization of English, Telugu and Hindi using BPE, WordPiece and SentencePiece.
Estimate cost per query from input and output token counts, context limits and model pricing.
Recommend which approach, or combination, suits different support needs.
The Three Approaches
1. Prompting

A ready-made LLM is guided using a system prompt (role, tone, rules) and a few example question-answer pairs.

Strengths: fastest and cheapest to set up, no training needed, easy to change.
Weaknesses: depends only on what the model already knows, so it can hallucinate or be out of date; long prompts repeat in every call and increase token cost.
2. Retrieval-Augmented Generation (RAG)

Company documents (FAQs, policies, product details) are split into chunks, converted into embeddings and stored in a vector database. For each customer query, the most relevant chunks are retrieved and added to the prompt so the model answers from real company data.

Strengths: grounds answers in documents, reduces hallucinations, handles new or changed information without retraining.
Weaknesses: needs a retrieval pipeline; retrieved text adds to the input tokens on every call; answer quality depends on retrieval quality.
3. Fine-Tuning

A small open-weight model is trained further on support question-answer pairs, using an efficient method such as LoRA, so it learns the domain, tone and answer format.

Strengths: consistent style, shorter prompts, can work well on repeated query types.
Weaknesses: needs good training data and compute, has a one-time training cost, and must be retrained when information changes.
	Prompting	RAG	Fine-Tuning
Setup effort	Low	Medium	High
Handles new information	Poorly	Well	Poorly (needs retraining)
Hallucination risk	Higher	Lower	Medium
Input token usage	Medium (long prompts)	Higher (retrieved context)	Lower
Extra cost	None	Vector DB and embeddings	Training

(The table is the starting expectation; the project's experiments will confirm or correct it.)

Tokenization and Multilingual Support

LLMs read tokens, not words. Text is split into tokens, each token is mapped to a token ID through a fixed vocabulary, and the model works on those IDs.

Common tokenization methods studied:

BPE (Byte Pair Encoding): repeatedly merges frequent character pairs into larger tokens.
WordPiece: builds subword units based on how likely they are in the training text.
SentencePiece: works directly on raw text without needing pre-split words, which suits many languages.

English text usually needs fewer tokens than Telugu or Hindi, whose scripts are often split into many more pieces. The same support message can therefore use more of the context window and cost more when written in an Indian language. This project measures that difference on real support queries.

Token Economics and Cost Estimation

Hosted models charge by tokens, and every model has a context limit (the most tokens it can handle in one request). The cost of one support conversation is estimated as:

cost = (input tokens × input price) + (output tokens × output price)

How each approach changes the cost:

Prompting: long system prompts and examples are repeated in every request.
RAG: retrieved passages are added to every request, increasing input tokens.
Fine-Tuning: prompts can be shorter, but there is a one-time training cost.
Methodology
Data preparation: collect sample support queries and a small knowledge base (FAQs, policies, product details) in English, Telugu and Hindi.
Tokenization analysis: tokenize the same queries with BPE, WordPiece and SentencePiece style tokenizers and record token counts and IDs for each language.
Prompting baseline: build the baseline with a system prompt and a few examples.
RAG pipeline: chunk the knowledge base, create embeddings, store them in a vector database and add retrieved passages to the prompt.
Fine-tuning: fine-tune a small open-weight model on question-answer pairs using LoRA.
Evaluation: run all three approaches on the same test queries and record the metrics below.
Cost estimation: calculate cost per query for each approach, including retrieved context and training cost.
Comparison: compare results across approaches and languages and write recommendations.
Evaluation Metrics
Answer accuracy: whether the answer is correct and matches the company information.
Hallucination rate: how often the answer includes made-up or unsupported details.
Failure rate: how often the system gives no answer, an error or an unusable answer.
Response time: how long each approach takes to reply.
Token usage: input and output tokens per query, per language.
Cost per query: estimated from token counts and model pricing.
Tech Stack
Tool	Purpose
Python 3	Main programming language
Jupyter Notebook / Google Colab	Running experiments, free GPU for fine-tuning
Hugging Face Transformers and Tokenizers	Open-weight models, tokenizers, fine-tuning
tiktoken, SentencePiece	Token counting
LLM API (closed model)	Prompting and RAG with a hosted model
FAISS or ChromaDB	Vector database for RAG
Sentence embedding model	Converting documents to embeddings
PEFT / LoRA	Low-cost fine-tuning
Pandas, Matplotlib	Data handling, cost tables and charts
Project Structure
├── data/            # sample support queries and knowledge base
├── tokenization/    # token count experiments (English, Telugu, Hindi)
├── prompting/       # prompting baseline
├── rag/             # RAG pipeline
├── finetuning/      # fine-tuning notebooks
├── evaluation/      # accuracy, hallucination and cost results
└── README.md
Setup and Usage
bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt

Run the notebooks in each folder in this order: tokenization → prompting → rag → finetuning → evaluation.

Expected Outcome
A working comparison of Prompting, RAG and Fine-Tuning for a customer support assistant.
Clear evidence of how each approach handles hallucinations, outdated knowledge and failures.
A token analysis showing how English, Telugu and Hindi differ in token counts and cost.
A cost estimate per query and per conversation for each approach.
A practical guideline on when to use Prompting, RAG, Fine-Tuning or a combination.
