![Profile views](https://komarev.com/ghpvc/?username=yuki-2025&label=Profile%20views) [![GitHub followers](https://img.shields.io/github/followers/yuki-2025?label=Followers&logo=github)](https://github.com/yuki-2025?tab=followers) ![Total Stars](https://img.shields.io/badge/dynamic/json?url=https://api.github-star-counter.workers.dev/user/yuki-2025&query=%24.stars&label=Stars&logo=github)

🤗 Hi, I’m @yuki-2025
-
I’m a seasoned LLM research engineer, ML engineer/data scientist, and AI product manager with training in Applied Data Science at the University of Chicago. I specialize in LLMs, large multimodal models, and AI agents. I bridge academia and industry—as Senior Staff at an AI company and as an AI researcher at UChicago Booth and the Data Science Institute—turning peer-reviewed research into production systems and leading end-to-end AI implementations.

💻 Expertise: AI Research & Large Language Models (LLM)🤖  • Large multimodal models (LMMs)🎵  • Machine Learning & Deep Learning📚   • Full-Stack GenAI Applications💡  • AI Agents🧠

My AI Projects (all open-sourced on GitHub):
-
1. [Paper: AgentNet: Dynamically Graph Structure Selection for LLM-Based Multi-Agent System](https://github.com/yuki-2025/Dyna_Swarm). 
   <details>  <summary>A dynamic, input-driven **multi-agent system (MAS)**, AgentNet, is introduced, executing over learned communication graphs (CoT, ToT, GoT). Advantage Actor–Critic (A2C) *reinforcement learning* is applied to learn a stable distribution over edges, and the base LLM (LoRA) is fine-tuned as a graph selector (**LLM-as-judge**)</summary>
    to choose the best topology per input. The approach achieves **state-of-the-art** (SOTA) performance on structured **reasoning** tasks (Crossword, Game-of-24, MMLU, BBH) and **code generation** (HumanEval), while maintaining latency comparable to CoT/ToT-style and static-swarm baselines. (Paper under review at EMNLP).</details>
2. [Paper: Medinotes: A Gen AI Framework for Medical Note Generation](https://github.com/yuki-2025/MediNotes).  
   MediNotes is a **first GenAI framework** that enhances clinical consultations by automating documentation and providing a **healthcare-domain–fine-tuned copilot** with retrieval-augmented generation (RAG), LLM and ambient listening. I developed and validated the system with clinicians at UChicago Medicine, culminating in **2 IEEE publications**.
3. [Paper: IntentVCNet: Bridging Spatio-Temporal Gaps for Intention-Oriented Controllable Video Captioning](https://github.com/thqiu0419/IntentVCNet) </br>
   IntentVCNet is a fine-tuned InternVL with LLaMA-Factory, earning second place in the [*IntentVC Challenge at ACM MM 2025*](https://www.aclweb.org/portal/content/intentvc-challenge-acm-mm-2025-intention-oriented-controllable-video-captioning) (Intention-Oriented Controllable Video Captioning), which resulted in a published ACM MM paper.
4. [mRAG: Multimodal RAG - Paper Q&A System + Evaluation](https://github.com/yuki-2025/mRAG) </br>
    To address the issues of traditional RAG systems—long processing times caused by OCR and text chunking, poor recall quality, and reliance on text-only embeddings—I built an embedding–retrieval–Q&A pipeline based on Qwen2.5VL’s **multimodal** capabilities, created a synthetic dataset, and evaluated the system using an **LLM as the judge**.
5. [NLP Research: Fine-Tuned LLM Embeddings for Business Insights](https://github.com/yuki-2025/embedding_project) </br> In collaboration with the University of Chicago Booth School of Business, I developed, fine-tuned, and optimized LLMs to generate high quality business-domain embeddings enriched with broad general knowledge—enabling the extraction of CEO-level actionable insights for management and financial decision-making.
<details>
  <summary>📂 More other projects and papers</summary>

   6. [Uchicago AI Hackathon 2024](https://github.com/yuki-2025/Ai-hackathon)
   Won 2nd place at the UChicago DSI AI Hackathon 2024 with a RAG medical Q&A chatbot. Built using **LangChain** for orchestration, **PostgreSQL** with vector embeddings for **hybrid search**, Streamlit for the front end, and **Google Cloud Vertex AI** to **fine-tune** and host **Llama 3-8B**, enabling secure access to patient records and general medical question answering.
   7. [Fine-Tuning Llama 3-8B for Structured Math Reasoning](https://github.com/yuki-2025/llama3-8b-fine-tuning-math) 
This project involves **fine-tuning Llama3 8b** to generate **JSON formats** for arithmetic questions and further post-process the output to perform calculations. This method incorporates the latest fine-tuning techniques such as **Qlora, Unsloth, and PEFT**. It enables faster training speeds and requires fewer computational resources.
   8. [AI Salesman](https://github.com/yuki-2025/RAG_projects/blob/main/Recommendation_LLM.ipynb)
   Built an AI-powered RAG hybrid search recommendation system using RAG that lets customers search products with filters like price. Implemented with LangChain, LLMs, and pgvector in PostgreSQL to segment product descriptions, generate embeddings, and deliver relevant recommendations..
   9. [Agentic RAG](https://github.com/yuki-2025/RAG_projects/blob/main/notebooks/en/agent_rag.ipynb)
      Built an Agentic RAG workflow with smolagents, wrapping retrieval as an agent tool for dynamic document search, compared against standard RAG (embedding + FAISS + LLM), and evaluated with LLM-as-a-Judge.
   
   11. Computer Vision (CV) collection <br> 
      ✦ [Style Transfer:](https://github.com/yuki-2025/cv_workshops/blob/main/style_transfer.ipynb) Implementing style transfer with TensorFlow/Keras <br>
      ✦ [MLflow:](https://github.com/yuki-2025/cv_workshops/blob/main/MLFlow.ipynb) Tutorial on using MLflow for experiment tracking <br>
      ✦ [Image Search RAG:](https://github.com/yuki-2025?page=2&tab=repositories) Image-based search system using RAG with Qdrant and Streamlit (search images by image input) <br>
      ✦ [Roboflow](https://github.com/yuki-2025/Roboflow): Step-by-step guide to annotating images and training a coin-detection model on Roboflow <br>
      ✦ [Aircraft Detection:](https://github.com/yuki-2025/CV_AircraftDetection) Training a YOLO model for military aircraft detection and model evaluation <br>
   
   12. Reproduced SOTA Research Papers
      ✦ Stanford Alpaca 7B – [dataset curation](https://github.com/yuki-2025/Reproduce_Paper/blob/main/DataMaker_for_Alpaca_style_custom_dataset.ipynb) and [instruction tuning of LLaMA](https://github.com/yuki-2025/Reproduce_Paper/blob/main/Alpaca_%2B_Llama_3_8b_full_example.ipynb) to achieve GPT-3.5-comparable performance. <br>
      ✦ [LLaVA](https://yuki-blog1.vercel.app/article/llava) – full training workflow to reproduce the multimodal model.<br>
      ✦ [LLaVA + RAG](https://github.com/yuki-2025/Reproduce_Paper/blob/main/Inference_with_LLaVa_for_multimodal_generation.ipynb) – semi-structured and multimodal retrieval-augmented generation. <br>
      ✦ [NanoGPT](https://github.com/yuki-2025/Reproduce_Paper/blob/main/gpt_dev.ipynb) – training a GPT model from scratch to understand Transformer internals. <br>
      ✦ [RAFT](https://github.com/yuki-2025/Reproduce_Paper/blob/main/RAFT_Finetuning_Starling7b.ipynb) – combining fine-tuning and RAG for improved retrieval performance. <br>

   13. Recommendation System <br>
      ✦ [Instacart Market Basket Analysis using PySpark](https://github.com/yuki-2025/recommendation-system/blob/main/Instacart-AssociationMining%20%281%29.ipynb) 
      Developed a scalable **market-basket analysis pipeline** on Instacart order data using **PySpar**k MLlib’s FPGrowth. Processed millions of transactions to extract **frequent itemsets** (≥1% support) and generated **association rules** (≥20% confidence, lift >1.5) for **co-purchase recommendations** (“customers who bought X also bought Y”).<br>
      ✦ [Collaborative Filtering Recommendation](https://github.com/yuki-2025/recommendation-system/blob/main/MovieRecommender%20%281%29.ipynb)
      Use PySpark to load and clean the data, train an **ALS model**, and Generate Top-10 movie recommendations for all users. Provide Top-10 recommendations for a specified subset of users. Identify the most likely users for a given set of movies.Make rating predictions and evaluate the model performance using RMSE.<br>
      ✦ [Two-Tower Recommendation System](https://github.com/yuki-2025/recommendation-system/blob/main/two_tower_final.ipynb)
        Use PySpark and **Spark SQL to clean**, join, and engineer user–item interaction features at scale. Encode movie titles with **SentenceTransformer** and load **user/item** metadata into **pandas** for downstream processing. Build and train a **Two-Tower neural network** in **PyTorch** that learns *user and item embeddings* via **contrastive loss**. Persist item embeddings in **Redis** as a vector database and leverage RedisVL for approximate **nearest-neighbor search** to return Top-K movie recommendations.<br>
   
   14. Useful apps and tools:
       - [Video_subtitle_generater:](https://github.com/yuki-2025/video_subtitle) Generate subtitles from an audio/video file, using OpenAI's Whisper model. Support multiple language.I take notes when learning from videos. It’s handy to have transcripts, and capturing that data is also useful for model training.
       - [Google Drive Helper:](https://github.com/yuki-2025/google_drive_helper ) The code I always use in my project when come to Google Cloud Platform. Instantly delete files, download them, edit permissions, and transfer ownership in bulk – all in just a few seconds.
       - [Blockchain apps:](https://github.com/yuki-2025/blockchain) 2 apps that run smart contracts and blockchain routes to demonstrate key blockchain principles: decentralization, immutability, Proof of Work (PoW), and transparency

</details>


## 🛠️ Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#) <!--Data Science & ML:<br> -->
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![R](https://img.shields.io/badge/R-%23276DC3.svg?logo=r&logoColor=white)](#)
[![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=fff)](#)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=fff)](#)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?logo=apachespark&logoColor=fff)](#)
[![LangChain](https://img.shields.io/badge/LangChain-1c3c3c.svg?logo=langchain&logoColor=white)](#)
[![GraphQL](https://img.shields.io/badge/GraphQL-E10098?logo=graphql&logoColor=fff)](#) 
[![FastAPI](https://img.shields.io/badge/FastAPI-009485.svg?logo=fastapi&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-000?logo=flask&logoColor=fff)](#)
[![Power BI](https://custom-icon-badges.demolab.com/badge/Power%20BI-F1C912?logo=power-bi&logoColor=fff)](#)
[![Tableau](https://custom-icon-badges.demolab.com/badge/Tableau-0176D3?logo=tableau&logoColor=fff)](#)
[![C#](https://custom-icon-badges.demolab.com/badge/C%23-%23239120.svg?logo=cshrp&logoColor=white)](#) <!-- Full stack - APP & Web : <br> -->
[![CSS](https://img.shields.io/badge/CSS-639?logo=css&logoColor=fff)](#)
[![Dart](https://img.shields.io/badge/Dart-%230175C2.svg?logo=dart&logoColor=white)](#)
[![Flutter](https://img.shields.io/badge/Flutter-02569B?logo=flutter&logoColor=fff)](#)
[![Go](https://img.shields.io/badge/Go-%2300ADD8.svg?&logo=go&logoColor=white)](#)
[![HTML](https://img.shields.io/badge/HTML-%23E34F26.svg?logo=html5&logoColor=white)](#) 
[![Swift](https://img.shields.io/badge/Swift-F54A2A?logo=swift&logoColor=white)](#)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=fff)](#)
[![Next.js](https://img.shields.io/badge/Next.js-black?logo=next.js&logoColor=white)](#)
[![NodeJS](https://img.shields.io/badge/Node.js-6DA55F?logo=node.js&logoColor=white)](#)
[![jQuery](https://img.shields.io/badge/jQuery-0769AD?logo=jquery&logoColor=fff)](#)
[![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?logo=chartdotjs&logoColor=fff)](#)
[![C](https://img.shields.io/badge/C-00599C?logo=c&logoColor=white)](#)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)
[![Java](https://img.shields.io/badge/Java-%23ED8B00.svg?logo=openjdk&logoColor=white)](#)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=000)](#)
[![JSON](https://img.shields.io/badge/JSON-000?logo=json&logoColor=fff)](#)
[![Solidity](https://img.shields.io/badge/Solidity-363636?logo=solidity&logoColor=fff)](#)
[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](#) <!-- cloud infra -->
[![AWS](https://custom-icon-badges.demolab.com/badge/AWS-%23FF9900.svg?logo=aws&logoColor=white)](#)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-%234285F4.svg?logo=google-cloud&logoColor=white)](#)
[![Microsoft Azure](https://custom-icon-badges.demolab.com/badge/Microsoft%20Azure-0089D6?logo=msazure&logoColor=white)](#)
[![Alibaba Cloud](https://img.shields.io/badge/AlibabaCloud-%23FF6701.svg?logo=alibabacloud&logoColor=white)](#)
[![Vercel](https://img.shields.io/badge/Vercel-%23000000.svg?logo=vercel&logoColor=white)](#)
[![Terraform](https://img.shields.io/badge/Terraform-844FBA?logo=terraform&logoColor=fff)](#)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](#)
[![GitLab CI](https://img.shields.io/badge/GitLab%20CI-FC6D26?logo=gitlab&logoColor=fff)](#)
[![Jenkins](https://img.shields.io/badge/Jenkins-D24939?logo=jenkins&logoColor=white)](#)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu&logoColor=white)](#)
[![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?logo=snowflake&logoColor=fff)](#)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?logo=databricks&logoColor=fff)](#)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff)](#)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?logo=kubernetes&logoColor=fff)](#)
[![ETL](https://custom-icon-badges.demolab.com/badge/ETL-9370DB?logo=etl-logo&logoColor=fff)](#) 
[![Figma](https://img.shields.io/badge/Figma-F24E1E?logo=figma&logoColor=white)](#) <!-- design -->
[![Canva](https://img.shields.io/badge/Canva-%2300C4CC.svg?&logo=Canva&logoColor=white)](#)
[![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?logo=mongodb&logoColor=white)](#)
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?logo=mysql&logoColor=fff)](#)
[![Neo4J](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)](#)
[![Oracle](https://custom-icon-badges.demolab.com/badge/Oracle-F80000?logo=oracle&logoColor=fff)](#)
[![Postgres](https://img.shields.io/badge/Postgres-%23316192.svg?logo=postgresql&logoColor=white)](#)
[![Redis](https://img.shields.io/badge/Redis-%23DD0031.svg?logo=redis&logoColor=white)](#)
[![SQLite](https://img.shields.io/badge/SQLite-%2307405e.svg?logo=sqlite&logoColor=white)](#)
[![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?logo=supabase&logoColor=fff)](#)
[![Teradata](https://img.shields.io/badge/Teradata-F37440?logo=teradata&logoColor=fff)](#)
[![ChatGPT](https://img.shields.io/badge/ChatGPT-74aa9c?logo=openai&logoColor=white)](#)      <!-- ai -->
[![Claude](https://img.shields.io/badge/Claude-D97757?logo=claude&logoColor=fff)](#)
[![Deepseek](https://custom-icon-badges.demolab.com/badge/Deepseek-4D6BFF?logo=deepseek&logoColor=fff)](#) 
[![GitHub Copilot](https://img.shields.io/badge/GitHub%20Copilot-000?logo=githubcopilot&logoColor=fff)](#)
[![Google Assistant](https://img.shields.io/badge/Google%20Assistant-4285F4?logo=googleassistant&logoColor=fff)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-886FBF?logo=googlegemini&logoColor=fff)](#)




<!---
yuki-2025/yuki-2025 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.

![My GitHub stats](https://github-readme-stats.vercel.app/api?username=yuki-2025&show_icons=true&theme=default&count_private=true) 
![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=yuki-2025&layout=compact&theme=default)
![GitHub Streak](https://github-readme-streak-stats.herokuapp.com/?user=yuki-2025&theme=default)
![GitHub Activity Graph](https://github-readme-activity-graph.vercel.app/graph?username=yuki-2025&theme=github)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
[![My Skills](https://skillicons.dev/icons?i=python,pytorch,sklearn,r,java,cpp,js,solidity,flask,react,html,css,postgres,mysql,mongodb,redis,docker,kubernetes,aws,gcp,azure,linux,git,vercel)](https://skillicons.dev)



--->
