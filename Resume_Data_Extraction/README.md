# Resume Data Extraction

A structured data extraction pipeline designed to parse raw, unstructured resume text (such as plain-text profiles, CVs, or OCR transcripts) into highly organized schema objects. This project uses **LangChain (LCEL)**, **Pydantic Validation**, and open-source models hosted via Hugging Face Inference Endpoints.

It is ideal for preprocessing professional applicant profiles, automating recruiter screening workflows, or normalizing candidate data for analytical systems.

## Features
- **Strict Parsing Enforcements:** Uses Pydantic to ensure all candidate details conform precisely to standardized programmatic data types.
- **LCEL Architecture:** Leveraging LangChain Expression Language for low-latency streaming and parallel batching support.
- **Hugging Face Agnostic Integration:** Works with open weights models (e.g., Llama 3.1, Mistral) via serverless endpoints, eliminating vendor lock-in.

## Schema Architecture
The extraction template isolates the following essential data fields from raw text:
- `candidate_name` (str): Full legal or professional name of the candidate.
- `skills` (list[str]): A list of core technical skills, programming languages, or methodologies explicitly or implicitly proven.
- `years_of_experience` (float/int): Quantifiable operational industry experience compiled from work history blocks.
- `highest_education` (str): Highest level of formal educational achievement (e.g., "B.S. Data Science", "Self-Taught").

## Setup Instructions

### 1. Navigate to Project
```bash
cd Langchain/Resume_Data_Extraction
