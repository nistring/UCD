"""ucd.llm against a stub OpenAI-compatible gateway (KHDP-like status codes); no network, no real key."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ucd import llm as L


class Stub(BaseHTTPRequestHandler):
    calls = []
    bodies = []          # request bodies of the POSTs
    script = []          # list of (status, body, headers) consumed per POST

    def log_message(self, *a):
        pass

    def _send(self, status, body, headers=None):
        raw = json.dumps(body).encode()
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        Stub.calls.append(self.headers.get("Authorization"))
        Stub.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        status, body, headers = Stub.script.pop(0)
        self._send(status, body, headers)

    def do_GET(self):
        Stub.calls.append(self.headers.get("Authorization"))
        self._send(200, {"data": [{"id": "chat-model"}, {"id": "qwen3-asr"}]})


@pytest.fixture
def server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    Stub.calls, Stub.script, Stub.bodies = [], [], []
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def _client(base, key):
    return L.LocalLLM({"llm": {"base_url": base, "model": "chat-model", "api_key": key}})


def test_retry_after_429_then_success(server):
    ok = {"choices": [{"message": {"content": "<think>x</think>{\"action\": \"status\"}"}}]}
    Stub.script = [(429, {"detail": "slow"}, {"Retry-After": "1"}), (200, ok, {})]
    assert L.extract_json(_client(server, "test_key").complete_text("s", "u")) == {"action": "status"}
    assert Stub.calls == ["Bearer test_key"] * 2


def test_no_credit_402_is_not_retried(server):
    Stub.script = [(402, {"error": "no_credit"}, {})]
    with pytest.raises(L.LLMError, match="credit"):
        _client(server, "k").complete_text("s", "u")
    assert len(Stub.calls) == 1


def test_key_file_and_pat_not_allowed_in_config(tmp_path, monkeypatch):
    f = tmp_path / "pat"
    f.write_text("khdp_pat_fromfile\n")
    monkeypatch.setenv("UCD_LLM_API_KEY_FILE", str(f))
    assert L._api_key({}) == "khdp_pat_fromfile"
    monkeypatch.delenv("UCD_LLM_API_KEY_FILE")
    monkeypatch.delenv("UCD_LLM_API_KEY", raising=False)
    with pytest.raises(L.LLMError):
        L._api_key({"api_key": "khdp_pat_inconfig"})


def test_models_lists_ids(server):
    assert _client(server, "k").models() == ["chat-model", "qwen3-asr"]


def test_extra_body_dropped_when_gateway_rejects_it(server):
    ok = {"choices": [{"message": {"content": "{\"action\": \"status\"}"}}]}
    Stub.script = [(400, {"error": "unknown field chat_template_kwargs"}, {}), (200, ok, {})]
    c = L.LocalLLM({"llm": {"base_url": server, "model": "chat-model", "api_key": "k",
                            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}})
    assert L.extract_json(c.complete_text("s", "u")) == {"action": "status"}
    assert c.extra_body == {} and len(Stub.calls) == 2


def test_khdp_base_reads_default_pat_file(tmp_path, monkeypatch):
    for v in ("UCD_LLM_API_KEY_FILE", "UCD_LLM_API_KEY", "KHDP_PAT"):
        monkeypatch.delenv(v, raising=False)
    f = tmp_path / "pat"
    f.write_text("khdp_pat_x\n")
    monkeypatch.setenv("KHDP_PAT_FILE", str(f))
    assert L._api_key({"base_url": "https://internal-api.khdp.net/v1", "api_key": "none"}) == "khdp_pat_x"
    assert L._api_key({"base_url": "http://localhost:8000/v1", "api_key": "none"}) == "none"


def test_key_placeholder_brackets_are_stripped(tmp_path, monkeypatch):
    f = tmp_path / "pat"
    f.write_text("<khdp_pat_abc>\n")
    monkeypatch.setenv("UCD_LLM_API_KEY_FILE", str(f))
    assert L._api_key({}) == "khdp_pat_abc"


def test_auto_falls_back_to_responses_api(server):
    ok = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"action\": \"status\"}"}]}]}
    Stub.script = [(404, {"detail": "use /v1/responses"}, {}), (200, ok, {})]
    c = _client(server, "k")
    assert L.extract_json(c.complete_text("s", "u")) == {"action": "status"} and c.api == "responses"


def test_verify_model_rejects_asr_and_unknown(server):
    c = L.LocalLLM({"llm": {"base_url": server, "model": "qwen3-asr", "api_key": "k", "verify_model": True}})
    with pytest.raises(L.LLMError, match="not a chat model"):
        c.complete_text("s", "u")
    c.model = "nope"
    with pytest.raises(L.LLMError, match=r"one of: \['chat-model'\]"):
        c.complete_text("s", "u")


def test_each_model_family_gets_its_own_thinking_switch(monkeypatch):
    """E-77: glm-5-3 ignored chat_template_kwargs, wrote its reasoning into the reply and was cut off at 6144 tokens;
    sent together with the generic switch, its own `thinking` field stopped working too."""
    cfg = {"base_url": "https://example.org/v1", "model": "m", "api_key": "none",
           "extra_body": {"chat_template_kwargs": {"enable_thinking": False, "other": 1}}}
    for var in ("UCD_LLM_THINKING", "UCD_PLANNER_THINKING", "UCD_PLANNER_MODEL", "UCD_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    glm = L.role_settings({**cfg, "roles": {"planner": {"model": "glm-5-3-ax"}}}, "planner")["extra_body"]
    assert glm == {"thinking": {"type": "disabled"}, "chat_template_kwargs": {"other": 1}}
    qwen = L.role_settings({**cfg, "roles": {"critic": {"model": "qwen3-8-max-ax", "thinking": True}}}, "critic")
    assert qwen["extra_body"]["chat_template_kwargs"] == {"other": 1, "enable_thinking": True}
    monkeypatch.setenv("UCD_PLANNER_THINKING", "1")
    assert L.role_settings({**cfg, "roles": {"planner": {"model": "glm-5-3-ax"}}}, "planner")["extra_body"]["thinking"] \
        == {"type": "enabled"}


def test_a_reply_cut_off_while_thinking_is_asked_again_without_thinking(server, monkeypatch):
    """E-79: with thinking on, the planner's reasoning used the whole 16k budget and the turn was lost."""
    monkeypatch.delenv("UCD_LLM_THINKING", raising=False)
    monkeypatch.delenv("UCD_PLANNER_THINKING", raising=False)
    cut = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
    ok = {"choices": [{"finish_reason": "stop", "message": {"content": "{\"action\": \"status\"}"}}]}
    Stub.script = [(200, cut, {}), (200, ok, {})]
    llm = L.LocalLLM({"llm": {"base_url": server, "model": "chat-model", "api_key": "k",
                              "roles": {"planner": {"thinking": True}}}})
    assert L.extract_json(llm.complete_text("s", "u")) == {"action": "status"} and llm.finish_reason == "stop"
    assert Stub.bodies[0]["chat_template_kwargs"]["enable_thinking"] is True
    assert Stub.bodies[1]["chat_template_kwargs"]["enable_thinking"] is False
