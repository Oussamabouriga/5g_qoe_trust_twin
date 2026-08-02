from dotenv import load_dotenv

from qoe_twin.openai_client import OpenAIExplanationClient

load_dotenv()

client = OpenAIExplanationClient()

response = client.generate_json(
"""
You must return ONLY valid JSON.

{
  "prediction":"test",
  "confidence":"high",
  "summary":"connection successful",
  "likely_causes":[],
  "recommended_operator_checks":[],
  "limitations":"none"
}
"""
)

print(response)