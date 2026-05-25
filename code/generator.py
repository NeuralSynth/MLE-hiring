import json
from llm import llm, clean_json_response
from config import API_SPECS_DIR, ROOT

# Load tool schemas once
def load_tool_schemas() -> str:
    schema_path = API_SPECS_DIR / "internal_tools.json"
    if schema_path.exists():
        try:
            return schema_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return "[]"

TOOL_SCHEMAS = load_tool_schemas()

GENERATOR_SYSTEM_PROMPT_NORMAL = f"""You are a customer support agent. Answer the customer's request using ONLY the retrieved support documentation provided.
Do not use your training knowledge. Do not make up product features or contact details. If the answer is not in the documents, state that you cannot answer the question and suggest escalating.

Do not echo PII (like full credit card numbers, SSNs, physical addresses, emails, or phone numbers) back in your response. Reference them generically instead (e.g., "your card ending in 8901").

If the customer is asking for an action (such as refund, subscription modification, password reset, or locking their account):
- Refer to the internal tools schema.
- If the action is destructive (like refunding money or deleting/modifying a subscription) and the customer's identity is NOT already verified in the conversation context, you MUST trigger the `verify_identity` tool instead of the requested action, and explain that they need to verify their identity.
- Otherwise, if identity is verified or the action is safe, generate the appropriate tool call in `actions_taken`.
- Conform exactly to the parameter types and required fields in the tool schemas.

Tool Schemas:
{TOOL_SCHEMAS}

Output only valid JSON matching this exact schema:
{{
  "response": "<response text>",
  "actions_taken": [
     {{
       "action": "<tool_name>",
       "parameters": {{ ... }}
     }}
  ],
  "source_documents": "<relative path of the single document used to answer the question, e.g. data/devplatform/screen/test.md, or empty string if none>"
}}

Output only the JSON object. No preamble, no explanation.
"""

GENERATOR_SYSTEM_PROMPT_ESCALATE = f"""You are a customer support agent. The customer's ticket has been flagged for escalation to a human representative.
Provide a polite, professional, and empathetic message informing the customer that their request has been escalated to a human specialist who will review their case and follow up shortly.

You MUST also trigger the `escalate_to_human` tool in the `actions_taken` array.
Choose the correct department ('billing', 'technical', 'security', 'legal', or 'general') and priority ('low', 'normal', 'high', 'urgent') based on the ticket context.

Tool Schemas:
{TOOL_SCHEMAS}

Output only valid JSON matching this exact schema:
{{
  "response": "<professional escalation message>",
  "actions_taken": [
     {{
       "action": "escalate_to_human",
       "parameters": {{
         "priority": "low" | "normal" | "high" | "urgent",
         "department": "billing" | "technical" | "security" | "legal" | "general",
         "summary": "<brief explanation of why escalation is required>"
       }}
     }}
  ],
  "source_documents": ""
}}

Output only the JSON object. No preamble, no explanation.
"""

def generate(ticket_text: str, chunks: list, escalate: bool) -> dict:
    if escalate:
        # Escalation path
        user_content = f"Customer Ticket:\n{ticket_text}"
        system_prompt = GENERATOR_SYSTEM_PROMPT_ESCALATE
    else:
        # Normal grounded generation path
        doc_context = "\n\n".join([f"Document {i+1} (Path: {doc['path']}):\n{doc['content']}" for i, doc in enumerate(chunks)])
        user_content = f"Retrieved Support Documentation:\n{doc_context}\n\nCustomer Ticket:\n{ticket_text}"
        system_prompt = GENERATOR_SYSTEM_PROMPT_NORMAL
        
    result = llm.complete(
        system=system_prompt,
        user=user_content
    )
    
    try:
        cleaned = clean_json_response(result)
        generated = json.loads(cleaned)
        
        # Post-validation
        response = generated.get("response", "").strip()
        actions = generated.get("actions_taken", [])
        if not isinstance(actions, list):
            actions = []
        source_doc = generated.get("source_documents", "").strip()
        
        # Verify source doc path exists relative to ROOT
        if source_doc:
            full_path = ROOT / source_doc
            if not full_path.exists() or not full_path.is_file():
                source_doc = "" # Clear if path does not exist to avoid citation penalty
                
        if escalate:
            actions = []
            source_doc = ""
            
        return {
            "response": response,
            "actions_taken": actions,
            "source_documents": source_doc
        }
    except Exception:
        # Fallback response
        if escalate:
            return {
                "response": "Your request has been escalated to a human support specialist. We will review your ticket and get back to you as soon as possible.",
                "actions_taken": [{
                    "action": "escalate_to_human",
                    "parameters": {
                        "priority": "normal",
                        "department": "general",
                        "summary": "Escalated by fallback policy."
                    }
                }],
                "source_documents": ""
            }
        else:
            return {
                "response": "Thank you for reaching out. We are looking into your request. A support representative will follow up with you shortly.",
                "actions_taken": [],
                "source_documents": ""
            }
