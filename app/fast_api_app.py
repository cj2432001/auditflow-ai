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

import contextlib
import os
import uuid
import datetime
import json
from collections.abc import AsyncIterator
from typing import Union, List, Optional

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging
from google.genai import types

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

load_dotenv()
setup_telemetry()

try:
    _, project_id = google.auth.default()
    logging_client = google_cloud_logging.Client()
    logger = logging_client.logger(__name__)
except Exception:
    import logging as std_logging
    logger = std_logging.getLogger(__name__)
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "local-project")

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPENSES_FILE = os.path.join(AGENT_DIR, "expenses.json")

# Helpers to load and save local expense audit trails
def load_expenses():
    if os.path.exists(EXPENSES_FILE):
        try:
            with open(EXPENSES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_expenses(expenses):
    try:
        with open(EXPENSES_FILE, "w") as f:
            json.dump(expenses, f, indent=2)
    except Exception as e:
        if hasattr(logger, "error"):
            logger.error(f"Failed to save expenses.json: {e}")
        else:
            print(f"Error: {e}")

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,
    lifespan=lifespan,
)
app.title = "capstone-project"
app.description = "Smart Travel & Expense Concierge API"


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    if hasattr(logger, "log_struct"):
        logger.log_struct(feedback.model_dump(), severity="INFO")
    else:
        logger.info(f"Feedback: {feedback.model_dump()}")
    return {"status": "success"}


# --- REST API Endpoints for Custom Dashboard ---

def check_for_interrupt(event) -> tuple[Optional[str], Optional[str]]:
    """Checks if an event from run_async contains an interrupt function call."""
    if event.content and event.content.parts:
        for p in event.content.parts:
            if p.function_call and p.function_call.name == "adk_request_input":
                # Extract message prompt from args
                msg = p.function_call.args.get("message", "") if p.function_call.args else ""
                return p.function_call.id, msg
    return None, None

@app.get("/api/expenses")
def get_expenses_history():
    """Retrieve history of all submitted expenses."""
    return load_expenses()

@app.post("/api/submit")
async def submit_expense(payload: dict = Body(...), request: Request = None):
    """Submits a new expense and executes the ADK workflow."""
    submitter = payload.get("submitter", "Anonymous")
    amount = float(payload.get("amount", 0.0))
    category = payload.get("category", "Other")
    description = payload.get("description", "")
    date = payload.get("date", datetime.date.today().isoformat())
    
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    user_id = "user"
    app_name = request.app.state.agent_app_name
    runner = request.app.state.runner
    
    # Ensure session exists
    try:
        await runner.session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
    except Exception:
        pass
        
    expense_data = {
        "amount": amount,
        "submitter": submitter,
        "category": category,
        "description": description,
        "date": date
    }
    input_payload = {"data": expense_data}
    
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=json.dumps(input_payload))]
    )
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    audit_trail = [
        f"[{time_str}] Received expense submission of ${amount:.2f} by {submitter}.",
        f"[{time_str}] Parsing expense fields..."
    ]
    
    status = "Pending Review"
    reviewed_by = "system"
    security_alert = False
    redacted_pii = []
    llm_review_data = None
    interrupt_message = None
    outcome_result = None
    
    try:
        from google.adk.agents.run_config import RunConfig, StreamingMode
        from app.agent import ApprovalOutcome, RiskReview, ExpenseDetails
        
        async for event in runner.run_async(
            new_message=message,
            user_id=user_id,
            session_id=session_id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE)
        ):
            ev_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check for interrupt event
            int_id, int_msg = check_for_interrupt(event)
            if int_id:
                interrupt_message = int_msg
                status = "Pending Review"
                audit_trail.append(f"[{ev_time}] Execution interrupted. Manual manager approval required.")
            
            # Extract intermediate node outputs
            if event.output is not None:
                output_val = event.output
                if isinstance(output_val, ExpenseDetails):
                    redacted_pii = output_val.redacted_pii
                    security_alert = output_val.security_alert
                    if redacted_pii:
                        audit_trail.append(f"[{ev_time}] Privacy Guard: Redacted SSN or Credit Card PII.")
                    else:
                        audit_trail.append(f"[{ev_time}] Privacy Guard: Passed. No PII scrubbed.")
                    if security_alert:
                        audit_trail.append(f"[{ev_time}] Threat Guard: Prompt injection signature identified!")
                elif isinstance(output_val, RiskReview):
                    llm_review_data = {
                        "risk_score": output_val.risk_score,
                        "risk_factors": output_val.risk_factors,
                        "recommendation": output_val.recommendation
                    }
                    audit_trail.append(
                        f"[{ev_time}] LLM Audit: Risk Score {output_val.risk_score}/10. Rec: {output_val.recommendation}."
                    )
                elif isinstance(output_val, ApprovalOutcome) or (isinstance(output_val, dict) and "approved" in output_val):
                    # Handle both dict and Pydantic object
                    if isinstance(output_val, dict):
                        approved_bool = output_val.get("approved")
                        reason_str = output_val.get("reason")
                        rev_by = output_val.get("reviewed_by")
                    else:
                        approved_bool = output_val.approved
                        reason_str = output_val.reason
                        rev_by = output_val.reviewed_by
                        
                    outcome_result = {
                        "approved": approved_bool,
                        "reason": reason_str,
                        "reviewed_by": rev_by
                    }
                    status = "Approved" if approved_bool else "Rejected"
                    reviewed_by = rev_by
                    audit_trail.append(f"[{ev_time}] Final decision: {status} by {reviewed_by}. Reason: {reason_str}")
        
        # Save records
        expense_record = {
            "session_id": session_id,
            "submitter": submitter,
            "amount": amount,
            "category": category,
            "description": description,
            "date": date,
            "status": status,
            "reviewed_by": reviewed_by,
            "security_alert": security_alert,
            "redacted_pii": redacted_pii,
            "audit_trail": audit_trail,
            "llm_review": llm_review_data,
            "interrupt_message": interrupt_message
        }
        
        history = load_expenses()
        history.insert(0, expense_record)
        save_expenses(history)
        
        return expense_record

    except Exception as e:
        err_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        audit_trail.append(f"[{err_time}] System Error: {str(e)}")
        
        # Save error record
        expense_record = {
            "session_id": session_id,
            "submitter": submitter,
            "amount": amount,
            "category": category,
            "description": description,
            "date": date,
            "status": "System Error",
            "reviewed_by": "system",
            "security_alert": security_alert,
            "redacted_pii": redacted_pii,
            "audit_trail": audit_trail,
            "llm_review": None,
            "interrupt_message": None
        }
        history = load_expenses()
        history.insert(0, expense_record)
        save_expenses(history)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/respond")
async def respond_to_interrupt(payload: dict = Body(...), request: Request = None):
    """Resumes the interrupted workflow with the user's manual decision."""
    session_id = payload.get("session_id")
    decision = payload.get("decision")  # "approve" or "reject"
    
    if not session_id or not decision:
        raise HTTPException(status_code=400, detail="Missing session_id or decision.")
        
    user_id = "user"
    app_name = request.app.state.agent_app_name
    runner = request.app.state.runner
    
    # Format the resume message with the function response payload
    # The response dict must map interrupt_id -> decision string so ADK
    # populates ctx.resume_inputs["human_decision"] as a plain string.
    resume_message = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="adk_request_input",
                    response={"human_decision": decision},
                    id="human_decision"
                )
            )
        ]
    )
    
    # Load history to update
    history = load_expenses()
    record = next((r for r in history if r["session_id"] == session_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Expense session not found.")
        
    res_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["audit_trail"].append(f"[{res_time}] Received manager decision: {decision.capitalize()}. Resuming workflow...")
    
    try:
        from google.adk.agents.run_config import RunConfig, StreamingMode
        from app.agent import ApprovalOutcome
        
        async for event in runner.run_async(
            new_message=resume_message,
            user_id=user_id,
            session_id=session_id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE)
        ):
            ev_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check for subsequent interrupt (not expected here, but for safety)
            int_id, int_msg = check_for_interrupt(event)
            if int_id:
                record["interrupt_message"] = int_msg
                record["audit_trail"].append(f"[{ev_time}] Execution interrupted again.")
                
            if event.output is not None:
                output_val = event.output
                if isinstance(output_val, ApprovalOutcome) or (isinstance(output_val, dict) and "approved" in output_val):
                    if isinstance(output_val, dict):
                        approved_bool = output_val.get("approved")
                        reason_str = output_val.get("reason")
                        rev_by = output_val.get("reviewed_by")
                    else:
                        approved_bool = output_val.approved
                        reason_str = output_val.reason
                        rev_by = output_val.reviewed_by
                        
                    record["status"] = "Approved" if approved_bool else "Rejected"
                    record["reviewed_by"] = rev_by
                    record["interrupt_message"] = None  # Cleared
                    record["audit_trail"].append(
                        f"[{ev_time}] Final decision: {record['status']} by {rev_by}. Reason: {reason_str}"
                    )
        
        save_expenses(history)
        return record
        
    except Exception as e:
        err_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record["status"] = "System Error"
        record["audit_trail"].append(f"[{err_time}] Resume Error: {str(e)}")
        save_expenses(history)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/expenses/clear")
async def clear_all_expenses():
    """Clears all expenses from local storage."""
    save_expenses([])
    return {"status": "success", "message": "Database cleared successfully."}



# --- Mount Frontend Dashboard Static Files ---
static_dir = os.path.join(AGENT_DIR, "app", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="static")


# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
