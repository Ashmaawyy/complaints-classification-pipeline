import logging
from typing import Optional
from langchain_openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field, ValidationError
from transformers import pipeline
from dotenv import load_dotenv
from pathlib import Path
import os

# Setup logging with emojis
logging.basicConfig(
    level=logging.INFO,
    format="🕒 %(asctime)s - 📍 %(name)s - [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load environment variables first
env_path = Path('.') / 'keys.env'
load_dotenv(env_path)

# Environment variable for OpenAI API
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define Pydantic schema
class ComplaintInfo(BaseModel):
    summary: str = Field(description="Brief summary of the complaint")
    sentiment: str = Field(description="Sentiment of the complaint (Positive, Neutral, Negative)")
    urgency: str = Field(description="Urgency level (Low, Medium, High)")

parser: BaseOutputParser = PydanticOutputParser(pydantic_object=ComplaintInfo)

# Define prompt
prompt = PromptTemplate(
    template=(
        "Analyze the following customer complaint and extract:\n"
        "1. A short summary\n"
        "2. The sentiment (Positive, Neutral, Negative)\n"
        "3. The urgency level (Low, Medium, High)\n\n"
        "Complaint: {complaint}\n\n{format_instructions}"
    ),
    input_variables=["complaint"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Define the OpenAI LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=150)

# Fallback Hugging Face model for sentiment
sentiment_model = pipeline("sentiment-analysis")

def analyze_complaint_with_openai(complaint: str) -> Optional[ComplaintInfo]:
    try:
        logging.info("🚀 Trying analysis using OpenAI LLM...")
        chain = prompt | llm | parser
        result = chain.invoke({"complaint": complaint})
        logging.info("✅ OpenAI LLM analysis succeeded!")
        return result
    except (ValidationError, Exception) as e:
        logging.warning(f"⚠️ OpenAI LLM failed: {e}")
        return None

def analyze_complaint_with_huggingface(complaint: str) -> ComplaintInfo:
    logging.info("🤖 Falling back to Hugging Face Transformers pipeline...")
    sentiment_result = sentiment_model(complaint)[0]

    sentiment = sentiment_result["label"]
    score = sentiment_result["score"]

    if sentiment == "POSITIVE":
        urgency = "Low"
    elif score > 0.85:
        urgency = "High"
    else:
        urgency = "Medium"

    logging.info(f"📝 Using basic heuristics for urgency: Sentiment={sentiment}, Score={score:.2f}")
    return ComplaintInfo(
        summary=complaint[:100] + ("..." if len(complaint) > 100 else ""),
        sentiment=sentiment.title(),
        urgency=urgency
    )

def classify_complaint(complaint: str) -> ComplaintInfo:
    logging.info("🔍 Starting complaint classification pipeline...")
    result = analyze_complaint_with_openai(complaint)
    if result:
        return result
    return analyze_complaint_with_huggingface(complaint)

# Main entry point
if __name__ == "__main__":
    sample_complaint = (
        "I’ve been trying to reach your customer support for a week with no response! "
        "My order is delayed and I’ve been charged twice. Extremely frustrating experience."
    )
    final_result = classify_complaint(sample_complaint)
    logging.info("🎉 Classification completed:")
    logging.info(final_result.model_dump_json(indent=2))
