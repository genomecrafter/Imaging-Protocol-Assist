# agents/fhir_converter.py
import os
import json
import re
from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError
from typing import Any, Dict

from . import fhir_models  # assumes this file is agents/fhir_models.py

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# initialize client - uses env var if you prefer, but explicit is fine
client = genai.Client(api_key=GEMINI_API_KEY)

# model to use; adjust if you prefer a different variant
MODEL_NAME = "gemini-2.5-flash"


def _call_genai_model(prompt: str) -> Any:
    """
    Call the GenAI SDK with compatibility for a few method names across versions.
    Returns the raw SDK response object.
    """
    models_service = getattr(client, "models", None)

    if models_service is not None:
        # Preferred modern method name
        gen_func = getattr(models_service, "generate_content", None)
        if gen_func is None:
            # older/alternate names observed in examples
            gen_func = getattr(models_service, "generate", None)
        if gen_func is not None:
            return gen_func(model=MODEL_NAME, contents=prompt)

    # fallback: older SDKs sometimes expose top-level generate
    gen_top = getattr(client, "generate", None)
    if gen_top is not None:
        # different arg names in older SDKs; try common ones
        try:
            return gen_top(model=MODEL_NAME, prompt=prompt)
        except TypeError:
            return gen_top(MODEL_NAME, prompt)

    raise RuntimeError(
        "No compatible generation method found on genai client. "
        "Please upgrade/downgrade google-genai to a supported version."
    )


def _extract_json_from_text(text: str) -> Dict:
    """
    Try robust JSON extraction from model text:
    - remove triple-backtick fences
    - try to json.loads directly
    - if fail, extract first {...} block
    """
    # remove leading/trailing ```json blocks
    text = text.strip()
    # remove code fences like ```json ... ```
    if text.startswith("```"):
        # drop the leading fence and trailing fence
        pieces = text.split("```")
        # pieces may be ['', 'json\n{...}', ''] or ['', '{...}', '']
        # find the piece that looks like JSON
        for piece in pieces:
            if piece.strip().startswith("{"):
                text = piece.strip()
                break

    # try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: find first { ... } substring
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # if still failing, raise with helpful debug text
    raise ValueError(f"Could not parse JSON from model output. Raw output snippet:\n{text[:1000]}")


def convert_with_gemini(final_json_path: str, output_path: str) -> Dict:
    """
    Convert final.json into a validated FHIR Bundle using Gemini (via google-genai SDK),
    validate with Pydantic schemas, save to file, and return the bundle dict.
    """
    # read input
    with open(final_json_path, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    prompt = f"""
Convert the following JSON into an OpenFHIR-compatible Bundle (R4) in JSON format.
- Use CarePlan for "recommendations" and "rationale".
- Use PlanDefinition for each "protocol_selection".
- Wrap everything in a Bundle (type=collection).
- Ensure resourceType, id, and required FHIR fields are included.
- Return only the JSON for the Bundle (no commentary).

Input JSON:
{json.dumps(final_data, indent=2)}
"""

    # call GenAI
    response = _call_genai_model(prompt)

    # get text
    raw_text = getattr(response, "text", None)
    if raw_text is None:
        # sometimes the SDK returns a dict-like object
        try:
            raw_text = response["text"]
        except Exception:
            raw_text = str(response)

    # extract JSON
    try:
        fhir_dict = _extract_json_from_text(raw_text)
    except ValueError as e:
        raise RuntimeError(f"Failed to parse GenAI output as JSON: {e}")

    # validate with pydantic
    try:
        bundle = fhir_models.FHIRBundle(**fhir_dict)
    except ValidationError as e:
        # include the validation error and a short snippet of the output for debugging
        snippet = json.dumps(fhir_dict)[:1200]
        raise RuntimeError(f"FHIR validation failed: {e}\n\nModel output snippet:\n{snippet}")

    # save validated JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle.dict(), f, indent=2, ensure_ascii=False)

    print(f"[FHIR] Bundle saved to {output_path}")
    return bundle.dict()


if __name__ == "__main__":
    # simple test runner (local)
    convert_with_gemini("final.json", "fhir_bundle.json")
