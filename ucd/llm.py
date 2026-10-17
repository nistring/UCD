"""Client for an OpenAI-compatible chat endpoint, normally the KHDP gateway's
hosted-model relay (credit-gated, PAT as the API key). Default (config.yaml): the KHDP internal gateway
`https://internal-api.khdp.net/v1` with one model per role (ucd/config.yaml). Use the host name, not an IP (an IP gives 404);
requests to internal-api.khdp.net bypass any HTTP proxy automatically (same effect as adding it to NO_PROXY).

Standard library only. Settings: ucd/config.yaml, overridden by environment variables:
  UCD_LLM_BASE_URL, UCD_LLM_MODEL
  UCD_LLM_API_KEY_FILE  path to a file holding the key (preferred for a KHDP PAT: never put a PAT in an exec
                        command or env inside a pod - exec requests are audit-logged - and never in config.yaml)
  UCD_LLM_API_KEY       the key itself (fine on your own machine)
  KHDP_PAT / KHDP_PAT_FILE   used for a khdp.net base URL when nothing above is set; KHDP_PAT_FILE defaults to
                        ~/.khdp_pat (a file is preferred: exec requests and their env are audit-logged)
  UCD_CRITIC_BASE_URL / UCD_CRITIC_MODEL   optional: a different endpoint/model for the clinical critic and
                        waiver judge (a judge from another model family is less biased toward the planner)

  UCD_LLM_API           chat | responses | auto (default auto: POST /chat/completions, and if the gateway answers
                        404/405/501 for it, POST /responses - the OpenAI Responses API that KHDP also relays)

The model id must exist in GET /v1/models and be a chat model (KHDP lists e.g. qwen3-asr, which only serves
/v1/audio/transcriptions). With `verify_model: true` (config.yaml) the first call checks this and stops with the
list of usable ids instead of failing later.

  python -m ucd.llm check     # base URL, model list, credit (KHDP), one tiny chat call; never prints the key
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path(__file__).resolve().parent / "config.yaml"
RETRY_STATUS = {429, 502, 503, 504}
NON_CHAT = ("asr", "audio", "transcri", "speech", "tts", "whisper", "embed", "rerank", "image")
_VERIFIED: set[tuple[str, str]] = set()     # (base_url, model) pairs already checked against /models
MAX_RETRIES = 5


class LLMError(RuntimeError):
    """Non-retryable failure (bad key, no credit, forbidden, bad request)."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def urlopen(req: urllib.request.Request, timeout: float):
    """urlopen that never routes the KHDP internal gateway through an HTTP proxy (the platform proxy rejects it),
    even when NO_PROXY was not set."""
    host = urllib.parse.urlsplit(req.full_url).hostname or ""
    if host.endswith("internal-api.khdp.net"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def load_config() -> dict[str, Any]:
    cfg = (yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}) if CONFIG.exists() else {}
    llm = cfg.setdefault("llm", {})
    llm["base_url"] = os.environ.get("UCD_LLM_BASE_URL", llm.get("base_url", "http://localhost:8000/v1"))
    llm["model"] = llm.get("model", "qwen")
    llm["api"] = os.environ.get("UCD_LLM_API", llm.get("api", "auto"))
    llm["api_key_config"] = llm.get("api_key", "none")   # kept so each role resolves its own key
    llm["api_key"] = _api_key(llm)
    return cfg


def _clean_key(k: str) -> str:
    """Strip whitespace, quotes and <...> left over from copying a placeholder such as '<PAT>'."""
    return str(k).strip().strip("<>\"' \t\r\n")


def _api_key(llm: dict[str, Any]) -> str:
    return _clean_key(_raw_api_key(llm))


def _raw_api_key(llm: dict[str, Any]) -> str:
    kf = os.environ.get("UCD_LLM_API_KEY_FILE") or llm.get("api_key_file")
    if kf:
        p = Path(os.path.expanduser(kf))
        if not p.exists():
            raise LLMError(f"API key file not found: {p}")
        return p.read_text(encoding="utf-8").strip()
    if "UCD_LLM_API_KEY" in os.environ:
        return os.environ["UCD_LLM_API_KEY"]
    key = llm.get("api_key", "none")
    if str(key).startswith("khdp_pat_"):
        raise LLMError("a KHDP PAT must not be stored in config.yaml; use UCD_LLM_API_KEY_FILE or ~/.khdp_pat")
    if "khdp.net" in str(llm.get("base_url", "")) and key in (None, "", "none"):
        if os.environ.get("KHDP_PAT"):
            return os.environ["KHDP_PAT"].strip()
        p = Path(os.path.expanduser(os.environ.get("KHDP_PAT_FILE", "~/.khdp_pat")))
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return key


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by Qwen-style models."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return re.sub(r"<think>.*\Z", "", text, flags=re.S).strip()     # cut off while still reasoning: no answer


_KEY_AHEAD = re.compile(r'\s*,\s*"[A-Za-z_][A-Za-z0-9_ \-]*"\s*:')     # , "next_key":
_END_AHEAD = re.compile(r"\s*([}\]]|$)")


def repair_json(s: str) -> str:
    """Fix what models get wrong inside JSON strings, mostly long SQL (E-25, E-34): raw newlines and tabs, and
    unescaped double quotes ("sofa"::INTEGER, "a", "b"). Context decides where a string ends: a key ends at a quote
    followed by ':'; a value inside an object ends only where the object continues (', "key":' or '}'); a value in
    an array at ',' / ']'. Used only after the strict parse failed."""
    out: list[str] = []
    stack: list[str] = []          # open containers: '{' or '['
    expect_key = False             # inside an object, before a key
    in_str, is_key, esc = False, False, False
    depth, sq = 0, False           # inside a string: open parentheses and SQL single quotes ("corr("a", "b")")
    i = 0
    while i < len(s):
        c = s[i]
        if not in_str:
            if c == '"':
                in_str, is_key = True, bool(stack) and stack[-1] == "{" and expect_key
                depth, sq = 0, False
            elif c in "{[":
                stack.append(c)
                expect_key = c == "{"
            elif c in "}]":
                if stack:
                    stack.pop()
                expect_key = False
            elif c == ",":
                expect_key = bool(stack) and stack[-1] == "{"
            elif c == ":":
                expect_key = False
            out.append(c)
        elif esc:
            out.append(c)
            esc = False
        elif c == "\\":
            out.append(c)
            esc = True
        elif c == '"' and (depth > 0 or sq) and not is_key:
            out.append('\\"')                     # an open "(" or "'" means this quote is SQL text
        elif c == '"':
            rest = s[i + 1:]
            if is_key:
                ends = bool(re.match(r"\s*:", rest))
            elif stack and stack[-1] == "{":
                ends = bool(_KEY_AHEAD.match(rest) or _END_AHEAD.match(rest))
            else:
                ends = bool(re.match(r"\s*[,\]]", rest) or _END_AHEAD.match(rest))
            if ends:
                in_str = False
                if is_key:
                    expect_key = False
                out.append('"')
            else:
                out.append('\\"')
        else:
            if c == "'":
                sq = not sq
            elif not sq and c == "(":
                depth += 1
            elif not sq and c == ")":
                depth = max(0, depth - 1)
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(c, c))
        i += 1
    return "".join(out)


def extract_json(text: str) -> Any:
    """First JSON object in the text (tolerates code fences, surrounding prose, raw newlines and unescaped quotes
    inside strings)."""
    t = strip_think(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.S) or re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    s = m.group(1) if m.re.groups else m.group(0)
    try:
        return json.loads(s)
    except ValueError as first:
        try:
            return json.loads(repair_json(s))
        except ValueError:
            raise first from None


# Appeals to authority (user, 2026-10-17): no model output may cite papers, authors, journals or "the literature" as
# evidence. Prompts forbid it; these patterns enforce it on what models write (final.md, notes, critic findings,
# intake fields). Named clinical definitions (KDIGO, Sepsis-3, SOFA) are concepts, not citations, and stay allowed.
NO_AUTHORITY = ("Never cite papers, authors, journals, guidelines-as-authority or 'the literature' as evidence (no "
                "'et al.', no author-year references, no DOIs or PMIDs, no 'studies show'). That is an appeal to "
                "authority. Justify every statement with the protocol, the data shown to you and explicit clinical or "
                "statistical reasoning. Named definitions (e.g. KDIGO stages, Sepsis-3, SOFA) may be used as definitions.")
_NAME = r"(?:(?:van|von|de|da|del|di|le|la)\s+)?[A-Z][a-z]{2,}[A-Za-z'\-]*"   # a surname: Vincent, O'Brien, van Walraven
# capitalised words that are concepts, units or calendar words, never authors ("KDIGO (2012)", "ICU (2019)", "Cox (1972)")
_NOT_AUTHOR = (r"(?!(?:January|February|March|April|May|June|July|August|September|October|November|December|Year|"
               r"Cox|Kaplan|Meier|Wald|Fisher|Student|Pearson|Spearman|Charlson|Elixhauser|Glasgow|Richmond|"
               r"Hb|Na|Cr|ICU|SOFA|SAPS|MELD|APACHE|KDIGO|Sepsis|MIMIC)\b)")
CITATION = re.compile(
    rf"\b{_NOT_AUTHOR}{_NAME}(?:\s+[A-Z]{{1,3}})?,?\s+(?:et\s+al\b\.?|and\s+colleagues\b|and\s+coworkers\b)"
    r"|\bet\s+al\b\.?"
    rf"|\(\s*{_NOT_AUTHOR}{_NAME}(?:\s+(?:and|&)\s+{_NAME})?(?:\s+et\s+al\.?)?,?\s+(?:19|20)\d{{2}}[a-z]?\s*\)"
    rf"|\b{_NOT_AUTHOR}{_NAME}\s+(?:et\s+al\.?\s+)?\((?:19|20)\d{{2}}[a-z]?\)"
    r"|\bdoi:\s*\S+|(?<![\d.])\b10\.\d{4,9}/\S{4,}|\bPMID:?\s*\d+|\bPMC\d{5,}"
    r"|\b(?:N\s?Engl\s?J\s?Med|NEJM|Lancet|JAMA|BMJ|Crit(?:ical)?\s+Care\s+Med(?:icine)?|Intensive\s+Care\s+Med"
    r"|Am\s+J\s+Respir\s+Crit\s+Care\s+Med|Ann\s+Intern\s+Med|Chest\s+\d{4};)\b"
    r"|(?i:\b(?:previous|prior|published|several|many|numerous)\s+(?:studies|trials|papers|reports)\s+(?:have\s+)?"
    r"(?:show(?:n|s|ed)?|report(?:ed|s)?|found|demonstrat(?:ed|es)|suggest(?:ed|s)?)\b)"
    r"|(?i:\b(?:the\s+)?literature\s+(?:shows|suggests|reports|supports|indicates)\b)"
    r"|(?i:\b(?:a|one|the)\s+(?:seminal|well-known|famous)\s+(?:study|trial|paper)\b)")


def citations(text: str, allowed: str = "") -> list[str]:
    """Appeals to authority in model-written text (empty when clean). A citation that also appears in `allowed`
    (the user's own request or protocol) is the user's, not the model's, and is not counted."""
    return [m.group(0).strip() for m in CITATION.finditer(text or "") if m.group(0).strip() not in (allowed or "")]


def strip_citations(text: str, allowed: str = "") -> str:
    """Model text with appeals to authority replaced by a marker (for text that is kept, e.g. critic findings);
    citations that also appear in `allowed` (the user's own words) are kept."""
    if not text:
        return text
    return CITATION.sub(lambda m: m.group(0) if m.group(0).strip() in (allowed or "") else "[citation removed]", text)


ROLES = ("planner", "intake", "critic")


def role_settings(llm: dict[str, Any], role: str) -> dict[str, Any]:
    """Settings for one agent role. Precedence (lowest -> highest):
    llm.* defaults < llm.roles.<role> (config.yaml) < UCD_LLM_MODEL (one model for every role) <
    UCD_<ROLE>_MODEL / UCD_<ROLE>_BASE_URL (e.g. UCD_CRITIC_MODEL). Thinking: UCD_LLM_THINKING < UCD_<ROLE>_THINKING
    (0/1); UCD_<ROLE>_MAX_TOKENS overrides the role's reply budget."""
    out = dict(llm)
    block = (llm.get("roles") or {}).get(role) or {}
    out.update({k: v for k, v in block.items() if v is not None and v != ""})
    if os.environ.get("UCD_LLM_MODEL"):
        out["model"] = os.environ["UCD_LLM_MODEL"]
    out["base_url"] = os.environ.get(f"UCD_{role.upper()}_BASE_URL", out["base_url"])
    out["model"] = os.environ.get(f"UCD_{role.upper()}_MODEL", out["model"])
    # the key belongs to the endpoint: a KHDP PAT is only ever sent to a khdp.net host
    if out["base_url"] != llm["base_url"] or "api_key" in ((llm.get("roles") or {}).get(role) or {}):
        out["api_key"] = _api_key({**out, "api_key": ((llm.get("roles") or {}).get(role) or {}).get(
            "api_key", llm.get("api_key_config", "none"))})
    host = urllib.parse.urlsplit(out["base_url"]).hostname or ""
    if str(out.get("api_key", "")).startswith("khdp_pat_") and not host.endswith("khdp.net"):
        out["api_key"] = "none"                     # never hand the KHDP token to another host
    think = out.get("thinking")
    for var in ("UCD_LLM_THINKING", f"UCD_{role.upper()}_THINKING"):    # all roles, then this role
        if os.environ.get(var) in ("0", "1"):
            think = os.environ[var] == "1"
    if os.environ.get(f"UCD_{role.upper()}_MAX_TOKENS", "").isdigit():
        out["max_tokens"] = int(os.environ[f"UCD_{role.upper()}_MAX_TOKENS"])
    eb = dict(out.get("extra_body") or {})
    kw = dict(eb.pop("chat_template_kwargs", None) or {})
    if "enable_thinking" in kw:                      # the config default is the generic switch: fold it into `think`
        think = kw.pop("enable_thinking") if think is None else think
        kw.pop("enable_thinking", None)
    if kw:
        eb["chat_template_kwargs"] = kw
    if think is not None:                            # each model family's own switch (E-77); a 400/422 drops it
        for k, v in thinking_switch(out["model"], bool(think)).items():
            eb[k] = {**eb[k], **v} if isinstance(v, dict) and isinstance(eb.get(k), dict) else v
    out["extra_body"] = eb
    return out


def thinking_switch(model: str, on: bool) -> dict[str, Any]:
    """The request field that turns reasoning on or off for this model. GLM ignores chat_template_kwargs and then
    writes its reasoning into the reply text (E-77); it honours its own `thinking` field. Qwen and DeepSeek use the
    chat-template switch."""
    if str(model).lower().startswith("glm"):
        return {"thinking": {"type": "enabled" if on else "disabled"}}
    return {"chat_template_kwargs": {"enable_thinking": on}}


class LocalLLM:
    def __init__(self, cfg: dict[str, Any] | None = None, timeout: int = 600, role: str = "planner"):
        llm = role_settings(dict((cfg or load_config())["llm"]), role)
        self.role = role
        self.base_url = llm["base_url"].rstrip("/")
        self.model = llm["model"]
        self.api_key = llm["api_key"]
        self.extra_body = llm.get("extra_body") or {}   # e.g. {"chat_template_kwargs": {"enable_thinking": false}}
        self.max_tokens = int(llm.get("max_tokens", 4096))
        self.temperature = float(llm.get("temperature", 0.2))
        self.timeout = timeout
        self.max_retries = MAX_RETRIES
        self.api = str(llm.get("api", "auto"))           # chat | responses | auto
        self.verify_model = bool(llm.get("verify_model", False))
        self.finish_reason: str | None = None

    def _request(self, method: str, url: str, body: dict | None = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, method=method,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.api_key}"})
            try:
                with urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:300]
                if e.code in RETRY_STATUS and attempt < self.max_retries:
                    wait = e.headers.get("Retry-After")
                    time.sleep(min(float(wait) if wait and wait.replace(".", "").isdigit() else 2 ** attempt * 2, 120))
                    continue
                hint = {401: "key missing/invalid/expired - reissue the PAT in KHDP account settings",
                        402: "no LLM credit left - top up in KHDP (check GET /me/credit)",
                        403: "not allowed for this account/project (see reason_code)",
                        404: "endpoint or model not found - check base URL and GET /v1/models",
                        422: "malformed request",
                        503: "gateway/auth service unavailable - check the PAT (must start with khdp_pat_), "
                             "else retry later and report the requestId"}.get(e.code, "")
                raise LLMError(f"HTTP {e.code} from {url.split('?')[0]}: {hint} {detail}", e.code) from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt * 2, 60))
                    continue
                raise LLMError(f"cannot reach {self.base_url}: {e}") from None

    def complete_text(self, system: str, user: str) -> str:
        t0 = time.time()
        try:
            return self._complete_text(system, user)
        except LLMError as e:
            from ucd import diag
            diag.error("llm", e, role=self.role, model=self.model, status=e.status, seconds=round(time.time() - t0))
            raise

    def _complete_text(self, system: str, user: str) -> str:
        if self.verify_model and (self.base_url, self.model) not in _VERIFIED:
            self.ensure_model()
        if self.api == "responses":
            return self._responses(system, user)
        try:
            return self._chat(system, user)
        except LLMError as e:
            if self.api != "auto" or e.status not in (404, 405, 501):
                raise
            try:                       # the gateway may relay this model only via the Responses API
                txt = self._responses(system, user)
            except LLMError:
                raise e from None
            self.api = "responses"
            return txt

    def _responses(self, system: str, user: str) -> str:
        body = {"model": self.model, "temperature": self.temperature, "max_output_tokens": self.max_tokens,
                "instructions": system, "input": [{"role": "user", "content": user}]}
        out = self._request("POST", f"{self.base_url}/responses", body)
        if isinstance(out.get("output_text"), str):
            return strip_think(out["output_text"])
        parts = [c.get("text", "") for item in out.get("output", []) if item.get("type") == "message"
                 for c in item.get("content", []) if c.get("type") in ("output_text", "text")]
        return strip_think("".join(parts))

    def _chat(self, system: str, user: str) -> str:
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                **self.extra_body}
        try:
            out = self._request("POST", f"{self.base_url}/chat/completions", body)
        except LLMError as e:
            # only a gateway/model that rejects the extension fields themselves loses them (E-44): dropping them after
            # any 400/422 silently turned reasoning on for the rest of the run (slow, cut-off replies)
            if e.status not in (400, 422) or not self.extra_body or not any(
                    k in str(e) for k in ("chat_template_kwargs", "enable_thinking", "thinking", "extra", "unexpected",
                                          "unknown")):
                raise
            from ucd import diag
            diag.event("error", "llm.extra_body_dropped", role=self.role, model=self.model, message=str(e)[:200])
            self.extra_body = {}
            body = {k: v for k, v in body.items() if k in ("model", "temperature", "max_tokens", "stream", "messages")}
            out = self._request("POST", f"{self.base_url}/chat/completions", body)
        choice = out["choices"][0]
        self.finish_reason = choice.get("finish_reason")     # "length" = cut off at max_tokens (E-37)
        return strip_think(choice["message"].get("content") or "")

    def model_entries(self) -> list[dict]:
        return list(self._request("GET", f"{self.base_url}/models").get("data", []))

    def models(self) -> list[str]:
        return [m.get("id") for m in self.model_entries()]

    @staticmethod
    def is_chat(entry: dict) -> bool:
        """Heuristic: an entry whose id or modality/type fields mention ASR, audio, embeddings etc. is not a chat model."""
        text = " ".join(str(entry.get(k, "")) for k in ("id", "modality", "modalities", "type", "task", "capabilities",
                                                         "endpoints", "object")).lower()
        return not any(w in text for w in NON_CHAT)

    def ensure_model(self) -> None:
        """Stop early with a clear message if the model id is not listed in /models or is not a chat model."""
        try:
            entries = self.model_entries()
        except LLMError as e:
            if e.status in (401, 402, 403):
                raise
            print(f"[llm] could not list models ({e}); continuing with {self.model!r}", file=sys.stderr)
            return
        chat = [m.get("id") for m in entries if self.is_chat(m)]
        entry = next((m for m in entries if m.get("id") == self.model), None)
        if entry is None or not self.is_chat(entry):
            why = "is not in GET /models" if entry is None else "is not a chat model"
            raise LLMError(f"model {self.model!r} {why}; set UCD_LLM_MODEL to one of: {chat}")
        _VERIFIED.add((self.base_url, self.model))

    def credit(self) -> Any:
        """KHDP only: remaining LLM credit (GET /me/credit on the gateway root)."""
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return self._request("GET", f"{root}/me/credit")


def check() -> int:
    cfg = load_config()
    llm = LocalLLM(cfg, timeout=120)
    llm.max_retries = 1            # a quick diagnosis, not minutes of back-off
    key = cfg["llm"]["api_key"]
    print(f"base_url : {llm.base_url}")
    print(f"model    : {llm.model}  (planner)")
    for r in ROLES[1:]:
        rs = role_settings(dict(cfg["llm"]), r)
        print(f"  {r:<8} : {rs['model']}" + (f"  thinking={rs['thinking']}" if rs.get("thinking") is not None else "")
              + ("" if rs["base_url"].rstrip("/") == llm.base_url else f"  @ {rs['base_url']}"))
    print(f"key      : {'set (' + key[:9] + '...)' if key and key != 'none' else 'none'}")
    if "khdp.net" in llm.base_url and not str(key).startswith("khdp_pat_"):
        print("WARNING  : the key does not start with khdp_pat_ - check the PAT file (no <>, quotes or spaces)")
    host = urllib.parse.urlsplit(llm.base_url).hostname or ""
    if re.fullmatch(r"[\d.]+", host) and not host.startswith("127."):
        print("WARNING  : base_url uses an IP address; the KHDP gateway needs the host name (an IP gives 404)")
    try:
        entries = llm.model_entries()
        for m in entries:
            extra = {k: m[k] for k in ("modality", "modalities", "type", "task", "capabilities", "endpoints") if k in m}
            print(f"models   : {m.get('id')}  {'chat' if llm.is_chat(m) else 'NOT chat'}  {extra or ''}")
        llm.ensure_model()
        print(f"model ok : {llm.model!r} is listed and looks like a chat model")
    except LLMError as e:
        print(f"models   : {e}")
        if "UCD_LLM_MODEL" in str(e):
            return 1
    if "khdp" in llm.base_url:
        try:
            print(f"credit   : {llm.credit()}")
        except LLMError as e:
            print(f"credit   : {e}")
    failed = 0
    for role in ROLES:                                 # one test call per role: each may use another model or switch
        if role != "planner" and role_settings(dict(cfg["llm"]), role)["model"] in (
                role_settings(dict(cfg["llm"]), r)["model"] for r in ROLES[:ROLES.index(role)]):
            continue                                   # same model already tested
        one = LocalLLM(cfg, timeout=120, role=role)
        one.max_retries, one.verify_model = 1, False
        try:
            t0 = time.time()
            r = one.complete_text("Reply with one JSON object only.", 'Return exactly {"action": "status"}')
            ok = extract_json(r) == {"action": "status"}
            print(f"chat     : {role:<8} {one.model}: {r[:100]!r} ({time.time() - t0:.1f}s, api={one.api}) -> JSON ok: {ok}")
            if one.finish_reason == "length":
                print(f"WARNING  : {one.model} hit its output limit on a one-line task: reasoning is probably still on")
            elif r.strip() and not r.lstrip().startswith(("{", "```")):
                print(f"WARNING  : {one.model} writes text before the JSON (inline reasoning?): its thinking switch "
                      f"{thinking_switch(one.model, False)} may not be honoured; long replies can be cut off")
            failed += not ok
        except Exception as e:
            print(f"chat     : {role:<8} {one.model}: FAILED {e}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(check() if sys.argv[1:] == ["check"] else (print(__doc__) or 0))
