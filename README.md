# Streamlit Deployment: scikit-learn RAG Assistant

Deployment status: **PREPARED / NOT PUBLISHED**

## Purpose

This folder is a deployment-only adaptation of the validated academic project. It replaces the Gradio interface with Streamlit while reusing the same RAG backend, configuration, prompts, routing, safety logic, memory behavior, processed knowledge base, and FAISS index. It does not replace or modify the frozen academic delivery package.

## Architecture and models

The browser interacts with `streamlit_app.py`, which calls the packaged backend in `Exercise_2_Backend/src/`. The backend embeds a query, searches the persistent FAISS index, builds grounded context, and generates an answer with the configured local LLM. Anonymous per-browser state is kept in `st.session_state`; expensive shared resources are cached with `st.cache_resource`. A process-wide lock serializes access to the shared cached pipeline while session IDs keep conversation buffers separate.

- LLM: `Qwen/Qwen2.5-0.5B-Instruct`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS, 645 packaged vectors and 645 metadata records
- Runtime memory: per-session rolling memory plus SQLite-backed session data
- Configuration: the unchanged validated values in `config.yaml`

## Local execution

Python 3.13 matches the validated environment. From this directory:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first grounded request may download Qwen and MiniLM from Hugging Face if they are not already cached.

## GitHub preparation

Use the **contents** of this folder as the repository root so that `streamlit_app.py`, `config.yaml`, and `requirements.txt` are at the top level.

```bash
git init
git add .
git status
git commit -m "Prepare Streamlit deployment"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Before committing, confirm that `git status` does not include `.env`, SQLite files, logs, caches, credentials, or model-cache files. Do not use a repository URL containing embedded credentials.

## Streamlit Community Cloud deployment

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that can read the repository.
2. Select **Create app**.
3. Choose the repository, the `main` branch, and `streamlit_app.py` as the main file path.
4. In advanced settings, select Python 3.13 if the option is available. Do not add secrets because this app does not require API credentials.
5. Deploy and monitor the build logs until dependency installation and startup complete.
6. Open the assigned URL and manually test one grounded question, one ambiguous question, clear/reset, and persistent-session deletion.
7. Record the real public URL only after the deployed application is verified.

No public Streamlit URL is claimed in this package.

## Memory and privacy

Each browser session receives a random anonymous UUID. Do not submit names, account data, health information, credentials, or other personal or sensitive identifiers. Redaction is best-effort and cannot identify every possible sensitive value.

SQLite is created at runtime under `Exercise_2_Backend/memory/` and is excluded from Git. **Clear conversation** removes the active rolling context and starts a new anonymous session without deleting the prior SQLite records. **Delete persistent session** removes the current session's rolling and SQLite data. Streamlit Community Cloud uses an ephemeral filesystem, so SQLite data and downloaded model caches may disappear after restarts, redeployments, or platform maintenance; this deployment does not promise durable persistence.

## Cold starts and deployment limitations

- The first request can be slow because MiniLM and Qwen may need to download and initialize.
- The CPU configuration intentionally preserves the validated model and generation settings; generation may be substantially slower than on a GPU.
- Qwen is loaded as float32 on CPU by the unchanged backend. The model, embedding model, FAISS index, and Streamlit runtime may approach Community Cloud memory limits.
- Community Cloud resource limits, outbound model-download availability, and sleep/restart behavior are controlled by the hosting platform.
- If the free host cannot load the unchanged model within its memory or startup limits, moving to a larger host is preferable to silently changing the validated model or academic settings.
- The application is limited to the packaged official scikit-learn material and can still produce incorrect or incomplete answers.
- Deterministic source availability does not establish semantic answer correctness; verify important guidance against the linked official documentation.

## Deployment-specific adaptation

Only the frontend adapter is new. The cached pipeline is protected by a frontend-level lock because Streamlit can serve concurrent sessions from shared resources. No backend code, model, prompt, retrieval setting, FAISS artifact, knowledge-base content, or academic evaluation metric was changed.

