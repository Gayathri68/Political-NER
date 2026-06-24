# Political-NER 🗳️🤖

A **Named Entity Recognition (NER)** system fine-tuned for the Indian political domain,
capable of identifying and classifying political entities across **multiple Indian languages**.
Built on a multilingual transformer model, this project extracts meaningful entities —
politicians, political parties, constituencies, and more — from multilingual political text.

---

## 📌 What is it?

Named Entity Recognition (NER) is the task of locating and classifying named entities
in text into predefined categories. Standard NER models trained on general English corpora
fail to capture the nuances of Indian political language and regional scripts.

This project uses the **Multilingual Pan-Indian Political Entity Recognition Model** to
identify political entities from Indian-language text, including:

- 🧑‍💼 **Politicians** — names of political figures
- 🏛️ **Political Parties** — party names (e.g., BJP, INC, AAP, DMK)
- 📍 **Constituencies** — electoral zones and geographic regions
- 📋 **Policies & Schemes** — government initiatives mentioned in context
- 🗺️ **Locations** — states, districts, and cities in political context

---

## 💡 Why is it useful?

India is a linguistically diverse democracy with news, debates, and political discourse
happening across Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam, and many more languages.

Standard NLP pipelines fail to handle:
- **Code-mixed text** (e.g., Hinglish)
- **Transliterated political names** across scripts
- **Domain-specific entities** unique to Indian elections and governance

This model bridges that gap — enabling downstream tasks like:
- Political news analytics and summarization
- Election monitoring and trend detection
- Social media political sentiment analysis
- Multilingual political knowledge graph construction

---

## 🛠️ Technologies Used

| Purpose               | Technology                                      |
|-----------------------|-------------------------------------------------|
| Language              | Python 3.8+                                     |
| NLP Framework         | Hugging Face Transformers                       |
| Base Model            | Multilingual BERT / XLM-RoBERTa (transformer)  |
| NER Tagging Format    | IOB2 (Inside-Outside-Beginning)                 |
| Notebook Environment  | Jupyter Notebook                                |
| Data Processing       | Pandas, NumPy                                   |
| Evaluation            | Seqeval (entity-level F1, Precision, Recall)    |

Political-NER/

└── Multilingual-Pan-Indian-Political-Entity-Recognition-Model-main/

├── data/                  # Annotated training and test datasets

├── models/                # Saved or fine-tuned model weights

├── notebooks/             # Jupyter notebooks for training & inference

├── src/                   # Core scripts for preprocessing and evaluation

├── requirements.txt       # Python dependencies

└── README.md

---

## ⚙️ How to Run

### Prerequisites

- Python 3.8 or higher
- pip
- Jupyter Notebook or JupyterLab

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Gayathri68/Political-NER.git
cd Political-NER/Multilingual-Pan-Indian-Political-Entity-Recognition-Model-main
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

Common dependencies include:

```bash
pip install transformers datasets seqeval torch pandas numpy jupyter
```

### Step 3 — Launch the Notebook

```bash
jupyter notebook
```

Open the relevant `.ipynb` file from the `notebooks/` folder and run all cells
from top to bottom to:

1. Load and preprocess the annotated dataset
2. Tokenize text with a multilingual tokenizer
3. Fine-tune or load the pre-trained NER model
4. Run inference on custom political text
5. Evaluate performance using entity-level F1 scores

### Step 4 — Run Inference on Custom Text

```python
from transformers import pipeline

ner = pipeline("ner", model="path/to/saved/model", aggregation_strategy="simple")

text = "Narendra Modi addressed a rally in Varanasi ahead of the Lok Sabha elections."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']} → {entity['entity_group']} (score: {entity['score']:.2f})")
```

**Sample Output:**

Narendra Modi  → POLITICIAN   (score: 0.98)

Varanasi       → CONSTITUENCY (score: 0.95)

Lok Sabha      → POLITICAL_BODY (score: 0.97)

---

## 📊 Entity Labels

| Label            | Description                              | Example                     |
|------------------|------------------------------------------|-----------------------------|
| `POLITICIAN`     | Name of a political person               | Rahul Gandhi, Mamata        |
| `PARTY`          | Political party name                     | BJP, INC, AAP, TMC          |
| `CONSTITUENCY`   | Electoral area or region                 | Varanasi, Thiruvananthapuram|
| `POLITICAL_BODY` | Legislature, parliament, or body         | Lok Sabha, Rajya Sabha      |
| `LOCATION`       | Geopolitical location in context         | Maharashtra, New Delhi      |
| `SCHEME`         | Government scheme or policy              | PM Kisan, MGNREGA           |

---

## 🌍 Supported Languages

The model is designed to operate on text across major Indian languages, including:

- Hindi (हिंदी)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Bengali (বাংলা)
- Kannada (ಕನ್ನಡ)
- Malayalam (മലയാളം)
- English (including code-mixed Hinglish)

---

## 📈 Future Improvements

- [ ] Web interface for live entity extraction
- [ ] Support for real-time news feed analysis
- [ ] Expanded entity types (alliances, election symbols)
- [ ] API endpoint for integration with news aggregators
- [ ] Larger annotated dataset across more regional languages

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request.

---

## 📬 Author

**Gayathri68** — [github.com/Gayathri68](https://github.com/Gayathri68)

---

*Built to make Indian political text more accessible and machine-readable across languages.*
---

## 📂 Project Structure

