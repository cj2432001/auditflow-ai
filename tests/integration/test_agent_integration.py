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

import json
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent, ApprovalOutcome


def test_agent_stream() -> None:
    """Integration test checking that the workflow runs and auto-approves a low expense."""
    session_service = InMemorySessionService()

    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    expense_data = {
        "amount": 45.00,
        "submitter": "Alice Smith",
        "category": "Meals",
        "description": "Team lunch",
        "date": "2026-07-06"
    }
    input_payload = {"data": expense_data}

    message = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(input_payload))]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one event"

    has_outcome = False
    for event in events:
        if event.output is not None:
            output = event.output
            if isinstance(output, ApprovalOutcome) or (isinstance(output, dict) and "approved" in output):
                has_outcome = True
                if isinstance(output, dict):
                    assert output.get("approved") is True
                else:
                    assert output.approved is True
                break
    assert has_outcome, "Expected workflow to output ApprovalOutcome"
