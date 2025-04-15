# 📄 Complaint Classifier Pipeline

This project demonstrates a simple end-to-end pipeline for summarizing customer feedback using **LangChain**, **Hugging Face Transformers**, and **Pydantic** for structured validation. It's designed with modularity in mind and can be easily integrated into cloud orchestration platforms like **Azure** and **Databricks**.

---

## 🔍 Objective
To process raw customer complaints and generate structured summaries that can be used for downstream analytics or to feed into LLM-powered systems.

---

## 🛠️ Tech Stack

| Tool/Library            | Purpose                                         |
|-------------------------|-------------------------------------------------|
| LangChain               | Orchestrating LLM workflows                    |
| Hugging Face Transformers | Text generation using open-source LLMs      |
| Pydantic                | Structured data validation                     |
| Transformers Pipeline   | Interface for loading and using LLMs          |
| Azure / Databricks      | (Optional) Orchestration and data integration |

---

## 🧠 Key Concepts

- **Prompt Engineering**: Customizing prompts to guide LLM outputs.
- **Text Summarization**: Converting raw feedback into concise summaries.
- **Schema Validation**: Ensuring model outputs adhere to a defined structure.
- **Modular Workflows**: Designed to plug into any orchestrator or CI/CD flow.

---

## 🔄 Workflow Overview

1. **Load Model**: Use Hugging Face's `flan-t5-base` model for summarization.
2. **Wrap with LangChain**: Use `HuggingFacePipeline` to interface with LangChain.
3. **Prompt Template**: Define the summarization instruction.
4. **Validate Output**: Use `PydanticOutputParser` to enforce schema.
5. **Orchestrate**: Simulate running this pipeline on Azure/Databricks.

---

## 📦 Example Input & Output

**Input:**
```
I ordered my laptop two weeks ago, and it still hasn't arrived. Customer service didn't help. Very disappointed.
```

**Output (Validated Schema):**
```json
{
  "summary": "Customer is unhappy due to delayed laptop delivery and unhelpful support.",
  "sentiment": "Negative",
  "Urgency": "High"
}
```

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/yourusername/complaint-classifier-pipeline
cd complaint-classifier-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
python main.py
```

---

## 🔧 Future Improvements

- Connect to Azure Data Lake for input/output data.
- Add LangChain Agents for more complex decision flows.
- Integrate classification labels (e.g., delivery issue, product quality).
- Add support for batch processing.

---

## 👤 Author
**Mohamed Ashmawi**  
MIS Analyst | Freelance Data Engineer  
Passionate about building AI-powered data pipelines that drive real insights.

---

## 📬 Feedback & Contributions
Feel free to fork, open issues, or contribute enhancements. Your feedback is welcome!
