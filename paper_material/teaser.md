+--------------------------------------------------------------------------------------------------------+
|                                     Figure 1: Overview of ScholarCopilot                               |
|                                                                                                        |
| +-------------------------------+   +----------------------------------------+   +-------------------+ |
| |      Traditional RAG          |   |           ScholarCopilot               |   | Evaluation Results| |
| +-------------------------------+   +----------------------------------------+   +-------------------+ |
| |                               |   | Input:                                 |   | Retrieval Acc.    | |
| | Input (Title & Abstract)      |   | ┌───────────────────────────────────┐  |   | ▓▓▓▓▓▓▓▓▓▓ 40%    | |
| |              ▼                |   | │Title: "Advancements of LLMs"      │  |   | ▓▓▓░░░░░░ 15%     | |
| | 🔍 Retriever (Fixed)          |   | │Abstract: "We survey recent..."    │  |   | ▓░░░░░░░░ 9%      | |
| |              ▼                |   | └───────────────────────────────────┘  |   |                   | |
| | Generation                    |   |                  ▼                    |   | Generation Quality| |
| | (Separate pipeline)           |   | Iterative Generation & Retrieval:      |   | ▓▓▓▓▓▓▓▓▓▓ 16.2   | |
| |                               |   | ┌───────────────────────────────────┐  |   | ▓▓▓▓▓▓▓▓▓░ 15.8   | |
| |                               |   | │Generation (iter. 1):              │  |   | ▓▓▓▓▓░░░░░ 14.4   | |
| |                               |   | │...GPT-4 <cite_start/> 🔍          │  |   |                   | |
| |                               |   | │⏳ Pause, trigger retrieval         │  |   +-------------------+ |
| |                               |   | └───────────────────────────────────┘  |                         |
| |                               |   |                  ▼                    |                         |
| |                               |   | Dense Retrieval (same model encoding):|                         |
| |                               |   | 🔍 retrieval token embedding ↔️ 📚 corpus embeddings           | |
| |                               |   |                  ▼                    |                         |
| |                               |   | 📄 Retrieved Citation                 |                         |
| |                               |   |                  ▼                    |                         |
| |                               |   | ┌───────────────────────────────────┐  |                         |
| |                               |   | │Insert citation & continue ✅      │  |                         |
| |                               |   | └───────────────────────────────────┘  |                         |
| |                               |   |                  ▼                    |                         |
| |                               |   | Generation (iter. 2)...               |                         |
| |                               |   | (Repeat iteratively ↻)                |                         |
| |                               |   |                  ▼                    |                         |
| |                               |   | Complete Introduction & Related ✅     |                         |
| |                               |   |                                        |                         |
| |                               |   | 🧩 Unified & Aligned Representation    |                         |
| +-------------------------------+   +----------------------------------------+                         |
|                                                                                                        |
| Figure 1: Given the paper's title and abstract as input, ScholarCopilot iteratively generates academic |
| Introductions and Related Work sections by dynamically triggering dense retrieval through unified      |
| retrieval tokens. Corpus embeddings and retrieval tokens are encoded by the same model, achieving      |
| representation alignment and significantly outperforming traditional RAG methods.                      |
+--------------------------------------------------------------------------------------------------------+