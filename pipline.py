import logging
import os
from typing import Optional
from langchain_openai import OpenAI
from langchain_litellm import Litellm
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field, ValidationError
from transformers import pipeline
from dotenv import load_dotenv
from pathlib import Path

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

# OpenAI LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=150)

# First Fallback Litellm LLM
litellm = Litellm(
    temperature=0,
    model="llama2-7b-chat",
    max_tokens=150,
    api_key=os.getenv("LLMA_LITELLM_API_KEY"),
)

# Second Fallback Hugging Face model for sentiment
sentiment_model = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f')

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

def analyze_complaint_with_litellm(complaint: str) -> Optional[ComplaintInfo]:
    try:
        logging.info("🚀 Trying analysis using Litellm LLM...")
        chain = prompt | litellm | parser
        result = chain.invoke({"complaint": complaint})
        logging.info("✅ Litellm LLM analysis succeeded!")
        return result
    except (ValidationError, Exception) as e:
        logging.warning(f"⚠️ Litellm LLM failed: {e}")
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
    try:
        result = analyze_complaint_with_openai(complaint)
        if result:
            return result
    except Exception as e:
        logging.warning(f"⚠️ OpenAI LLM failed: {e}")
    try:
        logging.info("🔄 OpenAI LLM failed, retrying with Litellm...")
        result = analyze_complaint_with_litellm(complaint)
        if result:
            return result
    except Exception as e:
        logging.warning(f"⚠️ Litellm LLM failed: {e}")
    
    # If both LLMs fail, fallback to Hugging Face
    logging.info("🔄 Both LLMs failed, falling back to Hugging Face...")
    try:
        result = analyze_complaint_with_huggingface(complaint)
        return result
    except Exception as e:
        logging.error(f"⚠️ Hugging Face analysis failed: {e}")
        return None

# Main entry point
if __name__ == "__main__":
    sample_complaint = (
        "I’ve been trying to reach your customer support for a week with no response! "
        "My order is delayed and I’ve been charged twice. Extremely frustrating experience."
    )
    final_result = classify_complaint(sample_complaint)
    logging.info("🎉 Classification completed:")
    logging.info(final_result.model_dump_json(indent=2))
