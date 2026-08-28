# %% [markdown]
# # EXP-0007 — Upstage Pro 4 train-only CoT generation
#
# This notebook-cell script sends only clean official-train questions to the
# teacher API. It never opens leaderboard/test files. The official train answer
# is kept local and is used only after generation for exact-match verification.

# %%
# Cell 1: install the OpenAI-compatible client if needed.
%pip install -q "openai>=1.81,<2"

# %%
# Cell 2: mount Drive, find the generated clean_train.csv, and load only it.
import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

try:
    from google.colab import drive
    MOUNT_ROOT = Path("/content/drive")
    if not (MOUNT_ROOT / "MyDrive").exists():
        drive.mount(str(MOUNT_ROOT))
    DRIVE_ROOT = MOUNT_ROOT / "MyDrive"
except ImportError:
    DRIVE_ROOT = Path("/content/drive/MyDrive")

assert DRIVE_ROOT.exists(), f"Drive not found: {DRIVE_ROOT}"

clean_candidates = [
    path for path in DRIVE_ROOT.rglob("clean_train.csv")
    if path.is_file()
]
if len(clean_candidates) != 1:
    print("Found clean_train.csv candidates:")
    for path in clean_candidates:
        print("-", path)
    raise RuntimeError(
        "Expected exactly one clean_train.csv. Upload the locally generated "
        "file to Drive, or remove duplicate copies."
    )

CLEAN_PATH = clean_candidates[0]
PROJECT_DIR = CLEAN_PATH.parent.parent
RUN_DIR = PROJECT_DIR / "runs" / "EXP-0007-upstage-pro4-train-cot"
RAW_DIR = RUN_DIR / "raw"
VERIFIED_DIR = RUN_DIR / "verified"
REPORT_DIR = RUN_DIR / "reports"
for path in [RAW_DIR, VERIFIED_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(CLEAN_PATH, dtype=str, keep_default_na=False)
train.columns = [str(column).strip() for column in train.columns]
required = {"id", "question", "answer"}
assert required.issubset(train.columns), train.columns.tolist()
train = train.fillna("")
train["id"] = train["id"].str.strip()
train["question"] = train["question"].str.strip()
train["answer"] = train["answer"].str.strip()

assert train["id"].is_unique
assert train["question"].ne("").all()
assert train["answer"].str.fullmatch(r"-?\d+").all()

print("Clean train:", CLEAN_PATH)
print("Clean train rows:", len(train))

# %%
# Cell 3: select a deterministic pilot. Change to None only after inspecting
# the 100-row results and confirming the API budget/rate limit.
SEED = 20260807
MAX_ROWS = 100

if MAX_ROWS is None:
    work = train.copy().reset_index(drop=True)
else:
    work = train.sample(
        n=min(MAX_ROWS, len(train)),
        random_state=SEED,
    ).reset_index(drop=True)

print("Rows for this run:", len(work))

# %%
# Cell 4: API key and teacher prompt.
# Rotate any key previously pasted into chat or a notebook first.
from openai import OpenAI

try:
    from google.colab import userdata
    UPSTAGE_API_KEY = userdata.get("UPSTAGE_API_KEY")
except Exception:
    UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY", "")

assert UPSTAGE_API_KEY, (
    "Save the newly generated key as the Colab Secret UPSTAGE_API_KEY "
    "or set the local UPSTAGE_API_KEY environment variable."
)

# Use the exact model ID shown in the Upstage console. The example supplied by
# the user uses solar-pro4; change only this value if the console names it differently.
MODEL = "solar-pro4"
BASE_URL = "https://api.upstage.ai/v1"
client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url=BASE_URL,
    timeout=180.0,
    max_retries=0,
)

# Prompt archive: this is the immutable first Upstage teacher prompt for
# EXP-0007 (`upstage_teacher_v1`). Later concise/retry wording must use a new
# prompt version instead of editing this text, so old runs remain reproducible.
TEACHER_SYSTEM_PROMPT = r"""
You are a meticulous mathematics teacher writing a supervised training trace.

Solve the problem independently. Do not rely on an external reference answer.
Use exact arithmetic and show the essential derivation, not just the result.
Then add a separate `Verification:` paragraph that checks the result by
substitution, reverse calculation, counting verification, units, or another
appropriate independent check.

If the problem requires a missing image, figure, diagram, table, or ambiguous
definition, do not guess. Return status `quarantine` and explain the reason.

Otherwise return status `solved`. The final line of `solution` must be exactly
`\boxed{INTEGER}`, where INTEGER is one signed base-10 integer.

Return JSON only, with this schema:
{
  "status": "solved" or "quarantine",
  "solution": "complete derivation with a Verification paragraph",
  "final_answer": "INTEGER or empty string",
  "reason": "empty when solved; explanation when quarantined"
}
""".strip()

def build_messages(question: str):
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{question}"},
    ]

# %%
# Cell 5: resumable API generation and local exact-match verification.
RAW_JSONL = RAW_DIR / "upstage_pro4_attempts.jsonl"
VERIFIED_CSV = VERIFIED_DIR / "upstage_pro4_verified_cot.csv"
QUARANTINE_CSV = VERIFIED_DIR / "upstage_pro4_quarantine.csv"
MANIFEST_JSON = REPORT_DIR / "upstage_pro4_manifest.json"

BOXED_RE = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")
INTEGER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])")

def normalize_integer(value):
    match = re.fullmatch(r"\s*(-?\d+)\s*", str(value or ""))
    return str(int(match.group(1))) if match else None

def parse_json_object(text: str):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start:end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None

def object_to_jsonable(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): object_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def response_usage(response):
    usage = getattr(response, "usage", None)
    value = object_to_jsonable(usage) or {}
    if not isinstance(value, dict):
        return {}
    return value

def call_teacher(question: str):
    last_error = None
    for attempt in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=build_messages(question),
                reasoning_effort="medium",
            )
            message = response.choices[0].message
            reasoning = str(getattr(message, "reasoning", None) or "")
            content = str(getattr(message, "content", None) or "")
            return {
                "ok": True,
                "response_id": str(getattr(response, "id", "") or ""),
                "reasoning": reasoning,
                "content": content,
                "usage": response_usage(response),
                "attempt": attempt,
            }
        except Exception as exc:
            last_error = repr(exc)
            if attempt < 5:
                time.sleep(min(60, 2 ** (attempt - 1)))
    return {"ok": False, "error": last_error, "attempt": 5}

def as_verified_row(row, api_result):
    if not api_result.get("ok"):
        return None, {
            "id": row["id"],
            "question": row["question"],
            "official_answer": row["answer"],
            "model_answer": "",
            "reason": "api_error: " + str(api_result.get("error", "unknown")),
            "teacher": MODEL,
            "verified": "false",
            "solution": "",
        }

    content = api_result.get("content", "")
    reasoning = api_result.get("reasoning", "")
    payload = parse_json_object(content)
    solution = str((payload or {}).get("solution", "") or content).strip()
    if reasoning and reasoning not in solution:
        solution = (reasoning.strip() + "\n\n" + solution).strip()
    model_answer = normalize_integer((payload or {}).get("final_answer", ""))
    if model_answer is None:
        boxed = BOXED_RE.findall(solution)
        model_answer = normalize_integer(boxed[-1]) if boxed else None
    status = str((payload or {}).get("status", "") or "").strip().casefold()
    reason = str((payload or {}).get("reason", "") or "").strip()

    if payload is None:
        reason = "invalid_json_response"
    elif status != "solved":
        reason = reason or "teacher_quarantine"
    elif model_answer != row["answer"]:
        reason = "teacher_answer_not_exact_match"
    elif not re.search(r"\\boxed\s*\{\s*-?\d+\s*\}\s*$", solution):
        reason = "missing_terminal_boxed_integer"
    else:
        return {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "solution": solution,
            "teacher": MODEL,
            "verified": "true",
            "model_answer": model_answer,
            "reasoning": reasoning,
        }, None

    return None, {
        "id": row["id"],
        "question": row["question"],
        "official_answer": row["answer"],
        "model_answer": model_answer or "",
        "reason": reason,
        "teacher": MODEL,
        "verified": "false",
        "solution": solution,
    }

def read_attempts():
    if not RAW_JSONL.exists():
        return []
    rows = []
    with RAW_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

attempts = read_attempts()
finished_ids = {
    row["id"] for row in attempts
    if row.get("status") in {"verified", "quarantine"}
}
pending = work.loc[~work["id"].isin(finished_ids)].copy()
started = time.time()

with RAW_JSONL.open("a", encoding="utf-8") as handle:
    for number, row in enumerate(pending.itertuples(index=False), start=1):
        api_result = call_teacher(row.question)
        verified_row, quarantine_row = as_verified_row(
            {"id": row.id, "question": row.question, "answer": row.answer},
            api_result,
        )
        if verified_row is not None:
            status = "verified"
            extracted = verified_row["model_answer"]
        else:
            status = "quarantine"
            extracted = (quarantine_row or {}).get("model_answer", "")
        record = {
            "id": row.id,
            "question": row.question,
            "official_answer": row.answer,
            "status": status,
            "extracted_answer": extracted,
            "verified_row": verified_row,
            "quarantine_row": quarantine_row,
            "api_result": api_result,
            "created_at_unix": time.time(),
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        completed = len(attempts) + number
        print(
            f"[UPSTAGE] {completed}/{len(work)} id={row.id} status={status} "
            f"elapsed={(time.time() - started) / 60:.1f}m",
            flush=True,
        )

# %%
# Cell 6: materialize verified/quarantine datasets and a reproducibility report.
attempts = read_attempts()
verified_rows = [row["verified_row"] for row in attempts if row.get("verified_row")]
quarantine_rows = [row["quarantine_row"] for row in attempts if row.get("quarantine_row")]

pd.DataFrame(verified_rows).to_csv(VERIFIED_CSV, index=False, encoding="utf-8")
pd.DataFrame(quarantine_rows).to_csv(QUARANTINE_CSV, index=False, encoding="utf-8")

def usage_total(field):
    return sum(
        int((row.get("api_result", {}).get("usage", {}) or {}).get(field, 0) or 0)
        for row in attempts
    )

manifest = {
    "experiment_id": "EXP-0007-upstage-pro4-train-cot",
    "teacher_model": MODEL,
    "base_url": BASE_URL,
    "input_source": str(CLEAN_PATH),
    "input_rows": len(work),
    "processed_rows": len(attempts),
    "verified_rows": len(verified_rows),
    "quarantine_rows": len(quarantine_rows),
    "prompt_version": "upstage_teacher_verified_cot_boxed_v1",
    "reasoning_effort": "medium",
    "official_evaluation_files_read": [],
    "usage": {
        "prompt_tokens": usage_total("prompt_tokens"),
        "completion_tokens": usage_total("completion_tokens"),
        "total_tokens": usage_total("total_tokens"),
        "reasoning_tokens": usage_total("reasoning_tokens"),
    },
    "outputs": {
        "raw_jsonl": str(RAW_JSONL),
        "verified_csv": str(VERIFIED_CSV),
        "quarantine_csv": str(QUARANTINE_CSV),
    },
}
MANIFEST_JSON.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
