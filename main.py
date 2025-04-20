import json
import os
from flask import Flask, request, jsonify
from decouple import config
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from typing import List

# Initialize Flask app
app = Flask(__name__)

# Define recommendation format
class GetRecommendationFormat(BaseModel):
    recommendation: List[str] = Field(
        description="Provide 5 actionable prevention steps for the patient's brain disease."
    )

# Function to get AI-based recommendations
def getRecommendationFromAI(data):
    os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

    PROMPT_TEMPLATE = """
    You are a specialized AI Neuroradiologist. Based on the patient's information and predicted brain disease from a deep learning model, 
    provide five personalized, preventive recommendations to help manage or mitigate the condition.

    Patient Information:
    - Age: {patient_age}
    - Gender: {patient_gender}
    - Symptoms: {symptoms}
    - Medical History: {medicalHistory}
    - Predicted Brain Disease: {brain_prediction}

    Return the recommendations in the following JSON format:
    {formatting_instructions}
    """

    llm = ChatOpenAI(temperature=0.5, model_name="gpt-4")
    parser = PydanticOutputParser(pydantic_object=GetRecommendationFormat)

    prompt = PromptTemplate(
        input_variables=[
            "patient_age", "patient_gender", "symptoms", 
            "medicalHistory", "brain_prediction",
            "formatting_instructions"
        ],
        template=PROMPT_TEMPLATE,
    )

    chain = prompt | llm | parser

    with get_openai_callback() as cb:
        result = chain.invoke(
            {
                "patient_age": data["patient_age"],
                "patient_gender": data["patient_gender"],
                "symptoms": data["symptoms"],
                "medicalHistory": data["medicalHistory"],
                "brain_prediction": data["brain_prediction"]["prediction"],
                "formatting_instructions": parser.get_format_instructions(),
            }
        )
        total_tokens = cb.total_tokens

    return {"recommendation": result.recommendation, "totalTokens": total_tokens}

# Flask route
@app.route("/get-recommendation", methods=["POST"])
def get_recommendation():
    try:
        data = request.json
        response = getRecommendationFromAI(data)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask server
if __name__ == "__main__":
    app.run(debug=True)
