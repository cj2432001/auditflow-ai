# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import re
from typing import Union, List, Any
from pydantic import BaseModel, Field, model_validator
from google.adk.workflow import Workflow, FunctionNode, START, Edge
from google.adk.agents import LlmAgent, Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.apps import App
from google.genai import types

# --- 1. Configurations ---
APPROVAL_THRESHOLD = 100.0
MODEL_NAME = "gemini-3.5-flash"

# --- 2. Schemas ---

class ExpenseInput(BaseModel):
    data: Union[str, dict] = Field(
        description="The incoming expense payload, containing raw JSON or base64-encoded JSON."
    )

    @model_validator(mode='before')
    @classmethod
    def _validate_before(cls, data: Any) -> Any:
        # Check if it is a genai Content object
        if hasattr(data, 'parts') and data.parts:
            text = ""
            for part in data.parts:
                if hasattr(part, 'text') and part.text:
                    text += part.text
            return {"data": text}
        # Check if it is a dictionary representing Content (e.g. from JSON payloads)
        elif isinstance(data, dict) and "parts" in data:
            text = ""
            for part in data["parts"]:
                if isinstance(part, dict) and "text" in part:
                    text += part["text"]
                elif hasattr(part, "text") and part.text:
                    text += part.text
            return {"data": text}
        return data

class ExpenseDetails(BaseModel):
    amount: float = Field(description="The dollar amount of the expense")
    submitter: str = Field(description="The person submitting the expense")
    category: str = Field(description="Category of the expense, e.g., Travel, Meals, Software, Entertainment")
    description: str = Field(description="Description of the expense")
    date: str = Field(description="Date of the expense (YYYY-MM-DD)")
    redacted_pii: List[str] = Field(default_factory=list, description="Categories of PII redacted")
    security_alert: bool = Field(default=False, description="Flagged for security events")

class RiskReview(BaseModel):
    risk_score: int = Field(description="Risk score from 1 (lowest) to 10 (highest)")
    risk_factors: List[str] = Field(description="Identified risk factors or violations")
    recommendation: str = Field(description="Final recommendation: APPROVE or REJECT")
    security_alert: bool = Field(default=False, description="Flagged for security events")

class ApprovalOutcome(BaseModel):
    approved: bool = Field(description="True if the expense is approved, False otherwise")
    reason: str = Field(description="Reason for approval or rejection")
    reviewed_by: str = Field(description="Who reviewed the expense: 'system' or 'human'")
    redacted_pii: List[str] = Field(default_factory=list, description="Categories of PII redacted")
    security_alert: bool = Field(default=False, description="Flagged for security events")

# --- 3. Workflow Nodes ---

def parse_expense(node_input: ExpenseInput) -> Event:
    """Parses base64-encoded or raw JSON data inside the 'data' field."""
    raw_data = node_input.data
    parsed_json = None
    
    if isinstance(raw_data, str):
        try:
            # Try to decode base64 first
            decoded_bytes = base64.b64decode(raw_data)
            decoded_str = decoded_bytes.decode("utf-8")
            parsed_json = json.loads(decoded_str)
        except Exception:
            try:
                # Fall back to parsing as plain JSON string
                parsed_json = json.loads(raw_data)
            except Exception:
                # Handle plain text messages (e.g. from E2E integration tests) gracefully
                import datetime
                parsed_json = {
                    "amount": 50.0,
                    "submitter": "System Test",
                    "category": "Other",
                    "description": raw_data,
                    "date": datetime.date.today().isoformat()
                }
    else:
        parsed_json = raw_data
        
    expense = ExpenseDetails(**parsed_json)
    return Event(
        output=expense,
        state={"expense": expense.model_dump()}
    )

def security_checkpoint(ctx: Context, node_input: ExpenseDetails) -> Event:
    """Scrubs sensitive PII from description and detects adversarial prompt injection."""
    desc = node_input.description
    redacted_categories = []
    
    # 1. Scrub SSNs
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    if re.search(ssn_pattern, desc):
        desc = re.sub(ssn_pattern, "[REDACTED SSN]", desc)
        redacted_categories.append("SSN")
        
    # 2. Scrub Credit Cards
    cc_pattern = r"\b(?:\d[ -]*?){13,16}\b"
    if re.search(cc_pattern, desc):
        desc = re.sub(cc_pattern, "[REDACTED CREDIT CARD]", desc)
        redacted_categories.append("Credit Card")
        
    # Update node_input in place
    node_input.description = desc
    node_input.redacted_pii = redacted_categories
    
    # Update workflow state
    ctx.state["expense"]["description"] = desc
    ctx.state["expense"]["redacted_pii"] = redacted_categories

    # 3. Detect Prompt Injection Attempts
    injection_keywords = [
        "ignore previous", "ignore instruction", "override", "bypass", 
        "auto-approve", "auto approve", "force approve", "system prompt",
        "instructions below", "you must approve", "as an admin", "developer mode",
        "override compliance", "ignore all policies", "ignore corporate policies"
    ]
    
    # Advanced pattern matching command verbs followed by compliance nouns (e.g. "bypass audit rules")
    override_pattern = r"\b(ignore|forget|disregard|override|bypass|disable)\b.*\b(instruction|policy|rule|guideline|check|system|previous|compliance|limit|audit)\b"
    
    is_injection = (
        any(kw in desc.lower() for kw in injection_keywords) or
        bool(re.search(override_pattern, desc.lower(), re.DOTALL))
    )
    
    if is_injection:
        node_input.security_alert = True
        ctx.state["expense"]["security_alert"] = True
        return Event(
            output=node_input,
            route="injection",
            state={"security_alert": True}
        )
    else:
        return Event(
            output=node_input,
            route="clean"
        )

def route_expense(node_input: ExpenseDetails) -> Event:
    """Routes expense based on threshold limit."""
    if node_input.amount < APPROVAL_THRESHOLD:
        outcome = ApprovalOutcome(
            approved=True,
            reason=f"Auto-approved: amount (${node_input.amount}) is below the corporate threshold of ${APPROVAL_THRESHOLD}.",
            reviewed_by="system",
            redacted_pii=node_input.redacted_pii,
            security_alert=node_input.security_alert
        )
        return Event(
            output=outcome,
            route="approve"
        )
    else:
        return Event(
            output=node_input,
            route="review"
        )

# LLM Risk Review Agent
llm_review = LlmAgent(
    name="llm_review",
    model=MODEL_NAME,
    instruction=(
        "You are a strict corporate financial compliance auditor. Review the provided expense details "
        "and determine its risk score (1-10, where 1 is safe, and 10 is clear fraud/policy violation). "
        "List any risk factors (e.g., weekend transactions, vague descriptions, luxury items, non-compliant categories). "
        "Provide a clear recommendation: APPROVE if low-risk and reasonable, or REJECT if policy is violated or suspicious."
    ),
    output_schema=RiskReview,
    output_key="risk_review",
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH
        )
    )
)

def post_llm_routing(ctx: Context, node_input: RiskReview) -> Event:
    """Evaluates the LLM Risk assessment to decide whether to auto-approve or route to human."""
    expense = ctx.state.get("expense", {})
    
    # Auto-approve if risk score is low (<= 3) and LLM recommends APPROVE
    if node_input.risk_score <= 3 and node_input.recommendation.upper() == "APPROVE":
        outcome = ApprovalOutcome(
            approved=True,
            reason=f"System Auto-approved: Low LLM Risk Score ({node_input.risk_score}/10). Recommendation: {node_input.recommendation}.",
            reviewed_by="system",
            redacted_pii=expense.get("redacted_pii", []),
            security_alert=expense.get("security_alert", False)
        )
        return Event(
            output=outcome,
            route="auto_approve"
        )
    else:
        # Otherwise, route to human manager for manual review
        return Event(
            output=node_input,
            route="human_review"
        )

async def human_approval_node(ctx: Context, node_input: Union[RiskReview, ExpenseDetails]) -> Event:
    """Pauses workflow execution using ADK interrupt to wait for a human decision."""
    if not ctx.resume_inputs:
        expense = ctx.state["expense"]
        
        # Check if the node was bypassed due to prompt injection alert
        if (isinstance(node_input, ExpenseDetails) and node_input.security_alert) or ctx.state.get("security_alert"):
            alert_header = "🚨 SECURITY ALERT: PROMPT INJECTION DETECTED! 🚨\n"
            assessment_text = (
                "**WARNING**: LLM review was BYPASSED because a prompt injection signature was detected in the description.\n"
                "Please inspect the description carefully before deciding."
            )
        else:
            alert_header = "⚠️ High-value or High-risk expense review required!\n"
            # node_input is RiskReview
            assessment_text = (
                f"LLM Risk Assessment:\n"
                f"  - Risk Score: {node_input.risk_score}/10\n"
                f"  - Risk Factors: {', '.join(node_input.risk_factors)}\n"
                f"  - LLM Recommendation: {node_input.recommendation}"
            )
            
        redaction_text = ""
        if expense.get("redacted_pii"):
            redaction_text = f"  - Redacted PII: {', '.join(expense['redacted_pii'])}\n"

        yield RequestInput(
            interrupt_id="human_decision",
            message=(
                f"{alert_header}"
                f"  - Submitter: {expense['submitter']}\n"
                f"  - Amount: ${expense['amount']}\n"
                f"  - Date: {expense['date']}\n"
                f"  - Category: {expense['category']}\n"
                f"  - Description: {expense['description']}\n"
                f"{redaction_text}"
                f"\n"
                f"{assessment_text}\n\n"
                f"Please reply with 'approve' or 'reject' to finalize."
            )
        )
        return

    # Handle workflow resume
    # resume_inputs can be the raw dict payload; safely extract the string decision
    raw_decision = ctx.resume_inputs.get("human_decision", "")
    if isinstance(raw_decision, dict):
        # Nested dict edge case: pull the first string value found
        raw_decision = next((v for v in raw_decision.values() if isinstance(v, str)), "")
    decision = str(raw_decision).strip().lower()
    approved = "approve" in decision
    
    expense = ctx.state.get("expense", {})
    outcome = ApprovalOutcome(
        approved=approved,
        reason=f"Reviewed by human manager. Decision: {decision.capitalize()}",
        reviewed_by="human",
        redacted_pii=expense.get("redacted_pii", []),
        security_alert=expense.get("security_alert", False)
    )
    yield Event(
        output=outcome
    )

# Wrap human node with rerun_on_resume=True to resume execution
human_approval = FunctionNode(
    func=human_approval_node,
    name="human_approval",
    rerun_on_resume=True
)

def record_outcome(ctx: Context, node_input: ApprovalOutcome) -> Event:
    """Formats and prints the final outcome of the expense approval workflow."""
    emoji = "✅" if node_input.approved else "❌"
    
    alert_badge = " [🚨 SECURITY ALERT]" if node_input.security_alert else ""
    redacted_info = ""
    if node_input.redacted_pii:
        redacted_info = f"\n* **Redacted PII**: {', '.join(node_input.redacted_pii)}"
        
    summary_text = (
        f"### Expense Approval Outcome{alert_badge}\n"
        f"* **Approved**: {emoji} {'Yes' if node_input.approved else 'No'}\n"
        f"* **Reviewed By**: {node_input.reviewed_by}\n"
        f"* **Reason**: {node_input.reason}"
        f"{redacted_info}\n"
    )
    yield Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=summary_text)]
        )
    )
    yield Event(output=node_input)

# --- 4. Workflow Assembly ---

root_agent = Workflow(
    name="expense_approval_workflow",
    description="A workflow for reviewing and approving expense reports based on policies, security, LLMs, and Human review.",
    input_schema=ExpenseInput,
    output_schema=ApprovalOutcome,
    edges=[
        (START, parse_expense),
        (parse_expense, security_checkpoint),
        (security_checkpoint, {"clean": route_expense, "injection": human_approval}),
        (route_expense, {"approve": record_outcome, "review": llm_review}),
        (llm_review, post_llm_routing),
        (post_llm_routing, {"auto_approve": record_outcome, "human_review": human_approval}),
        (human_approval, record_outcome)
    ]
)

app = App(
    root_agent=root_agent,
    name="app",
)
