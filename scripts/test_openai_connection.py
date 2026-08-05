from dotenv import load_dotenv

from qoe_twin.openai_client import OpenAIExplanationClient

load_dotenv()

client = OpenAIExplanationClient()

response = client.generate_json(
"""
You must return ONLY valid JSON.

{
  "prediction":"future_poor_qoe",
  "confidence":"low",
  "summary":"Insufficient evidence is available in this connection test.",
  "likely_causes":[],
  "recommended_operator_checks":["Review supplied operational evidence."],
  "limitations":"Insufficient evidence supports a specific cause."
}
"""
)

print(response)
