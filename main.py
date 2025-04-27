import json
import logging
import requests
from flask import Flask, request, jsonify
from decouple import config
from pydantic import BaseModel, Field
from typing import List

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = config("GEMINI_API_KEY", default="AIzaSyAWg1Vk9_WUDuehLy-ojdKLht6kOpuftdE")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

class GetRecommendationFormat(BaseModel):
    recommendation: List[str] = Field(
        description="Provide 5 actionable prevention steps for the patient's brain tumor disease."
    )

def build_prompt(data):
    return f"""
You are a specialized AI Neuroradiologist. Based on the patient's information and predicted brain tumor disease from a deep learning model, 
provide five personalized, preventive recommendations to help manage or mitigate the condition.

- Age: {data['patient_age']}
- Gender: {data['patient_gender']}
- Symptoms: {data['symptoms']}
- Medical History: {data['medicalHistory']}
- Predicted Brain Disease: {data['brain_prediction']['prediction']}

Return the output as a JSON list of 5 actionable recommendations under the key "recommendation".
Example:
{{ "recommendation": ["tip1", "tip2", "tip3", "tip4", "tip5"] }}
""".strip()

def getRecommendationFromGemini(data):
    prompt = build_prompt(data)
    logging.info("🚀 Sending prompt to Gemini:\n%s", prompt)

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)

    if response.status_code != 200:
        logging.error("❌ Gemini API error: %s", response.text)
        raise Exception(f"Gemini API error: {response.text}")

    candidates = response.json().get("candidates", [])
    if not candidates:
        raise Exception("No response candidates from Gemini API.")

    raw_text = candidates[0]["content"]["parts"][0]["text"]
    logging.info("🎯 Gemini raw response:\n%s", raw_text)

    try:
        # Strip triple backticks and parse JSON
        clean_text = raw_text.strip("`").replace("json", "").strip()
        recommendation_data = json.loads(clean_text)
        GetRecommendationFormat(**recommendation_data)  # Validate structure
        return recommendation_data
    except Exception as e:
        logging.exception("❌ Failed to parse Gemini response")
        raise Exception(f"Failed to parse Gemini response: {e}")

@app.route("/get-recommendation", methods=["POST"])
def get_recommendation():
    try:
        data = request.json
        logging.info("📥 Incoming request data: %s", data)
        response = getRecommendationFromGemini(data)
        return jsonify(response), 200
    except Exception as e:
        logging.exception("🔥 Exception in /get-recommendation")
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(debug=True)
