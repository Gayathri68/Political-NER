# Multilingual Pan-Indian Political Entity Recognition Model

## Project Overview

This project presents a **Multilingual Pan-Indian Political Entity Recognition (NER) System** designed to identify political entities such as **political parties, political leaders, government-related terms, and election-related entities** from multilingual Indian news text.

The system follows a complete end-to-end NLP pipeline consisting of:

- Web scraping / data collection
- Text preprocessing and cleaning
- Weakly supervised labeling
- Transformer-based model training and testing
- Prediction and evaluation report generation

The project is implemented across **12 Indian languages**, making it a multilingual political domain NLP system for large-scale political entity recognition.

The transformer-based models used in this project are:

- **MuRIL** (Multilingual Representations for Indian Languages)
- **XLM-R** (XLM-RoBERTa)

---

## Objective

The main objective of this project is to build a **multilingual political entity recognition system** that can automatically extract political entities from Indian news articles across multiple Indian languages using a weakly supervised and transformer-based approach.

---

## Workflow / Methodology

### Step 1: Data Collection / Web Scraping

News data is collected for multiple Indian languages through web scraping and stored in structured JSONL format.

**Folder:** `Output_files_step1/`

**Purpose:**
- Collect multilingual political news articles
- Store scraped news content in JSONL format
- Build the raw multilingual dataset for further processing

**Related notebooks:**
- `WebScraping_Code_1.ipynb`
- `Webscraping_code_2.ipynb`

---

### Step 2: Text Preprocessing

The collected multilingual news text is cleaned and preprocessed by removing noise, normalizing text, and preparing it for weak labeling and model training.

**Script used:** `step2_preprocess.py`

**Folder:** `output_files_step2/`

**Purpose:**
- Clean multilingual text data
- Normalize and standardize content
- Prepare the data for weak labeling and transformer training

---

### Step 3: Weakly Supervised Labeling

A weak labeling approach is applied to automatically assign NER labels to the cleaned multilingual text. This step creates labeled training data for political entity recognition.

**Script used:** `step3_ner_label.py`

**Folder:** `output_files_step3/`

**Purpose:**
- Generate weakly labeled NER annotations
- Identify political entities in multilingual news text
- Create labeled data for model training

**Visualization outputs:** `step3_Output/`

---

### Step 4: Model Training and Testing

The labeled multilingual dataset is used to train and evaluate transformer-based NER models.

**Main script used:** `Train_Test.py`

**Models used:**
- **MuRIL**
- **XLM-R**

This stage includes:
- Tokenization
- Label alignment
- Fine-tuning
- Validation
- Testing
- Prediction generation
- Comparative model evaluation

---

## Technologies Used

| Category | Tools / Frameworks |
|---|---|
| Language | Python |
| Notebooks | Jupyter Notebook |
| Deep Learning | PyTorch |
| NLP Models | Hugging Face Transformers, MuRIL, XLM-R |
| Data Format | JSONL |
| Techniques | Weak Supervision, Multilingual NLP, Named Entity Recognition (NER) |

---

## Key Features

- Supports **12 Indian languages**
- End-to-end multilingual NLP pipeline
- Political domain-specific entity recognition
- Weakly supervised NER data creation
- Transformer-based multilingual model training
- Comparative analysis of **MuRIL vs XLM-R**
- Prediction outputs and evaluation reports
- Organized project structure for reproducibility

---

## Repository Structure

```
Multilingual-Pan-Indian-Political-Entity-Recognition-Model/
│
├── Images/                          # Project screenshots / output images
├── Output_files_step1/              # Step 1: Scraped multilingual news data
├── output_files_step2/              # Step 2: Preprocessed multilingual data
├── output_files_step3/              # Step 3: Weakly labeled multilingual data
├── project_outputs/                 # Logs, predictions, reports, and final outputs
│   ├── logs/
│   ├── predictions/
│   └── reports/
├── step3_Output/                    # Weak labeling output visualizations
│
├── WebScraping_Code_1.ipynb         # Web scraping notebook 1
├── Webscraping_code_2.ipynb         # Web scraping notebook 2
├── step2_preprocess.py              # Preprocessing script
├── step3_ner_label.py               # Weak labeling script
├── Train_Test.py                    # Model training and testing script
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore rules
```

---

## How to Run the Project

### Step 1: Web Scraping / Data Collection

Run the following notebooks in order:

```
WebScraping_Code_1.ipynb
Webscraping_code_2.ipynb
```

### Step 2: Text Preprocessing

```bash
python step2_preprocess.py
```

### Step 3: Weak Labeling

```bash
python step3_ner_label.py
```

### Step 4: Model Training and Testing

```bash
python Train_Test.py
```

---

## Conclusion

This project provides a complete multilingual NLP framework for recognizing political entities from Indian news articles. By combining **web scraping**, **text preprocessing**, **weak supervision**, and **transformer-based multilingual NER models**, the system enables efficient political entity extraction across **12 Indian languages**.

---

## Author

**Varsha Janaki**
Final Year Project — Multilingual Pan-Indian Political Entity Recognition Model
