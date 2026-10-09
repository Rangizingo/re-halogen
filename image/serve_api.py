#!/usr/bin/env python3
# halogen-flash-server: source shipped as-is, comments stripped. The README is the documentation.
import argparse
import array
import asyncio
import base64
import binascii
import collections
import io
import math
import itertools
import json
import os
import random
import sys
import time
import uuid
import zlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_parse import (TC_CLOSE, TC_OPEN, ToolStream,
                        normalize_messages, split_tool_calls)

import request_records

SCHEMA_PREFACE = os.environ.get("HALOGEN_SCHEMA_ESCAPE", "") not in ("argmax", "off", "0", "none", "plain")
SSE_PROF = {"t": 0.0, "n": 0}
MODEL_ID = os.environ.get("HALOGEN_MODEL_ID", "halogen-qwen3.8-flash-next")

ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
               403: "permission_error", 404: "not_found_error",
               429: "rate_limit_error"}

VIS_PATCH = 16
VIS_MERGE = 2
VIS_UNIT = VIS_PATCH * VIS_MERGE
VIS_MIN_PIXELS = 256 * 256
VIS_MAX_PIXELS = int(os.environ.get("HALOGEN_VISION_MAX_PIXELS", 2560 * 1440))
API_VERSION = (os.environ.get("HALOGEN_IMAGE_VERSION") or "unknown").strip() \
    or "unknown"
SNAP3_ON = os.environ.get("HALOGEN_CACHE_SNAP3", "1").strip() != "0"
IMAGE_TOKEN_ID = None
VIDEO_TOKEN_ID = None
VISION_START_ID = None
VISION_END_ID = None
PAD_TEXT_IDS = {}

def pillow_ready():
    global _PILLOW
    if _PILLOW is None:
        try:
            from PIL import Image
            _PILLOW = True
        except Exception:
            _PILLOW = False
    return _PILLOW

_PILLOW = None

def vis_resolve_token(tok):
    global IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, VISION_START_ID, VISION_END_ID

    def one(name):
        i = tok.convert_tokens_to_ids(name)
        return i if isinstance(i, int) and i >= 0 else None
    IMAGE_TOKEN_ID = one("<|image_pad|>")
    VIDEO_TOKEN_ID = one("<|video_pad|>")
    VISION_START_ID = one("<|vision_start|>")
    VISION_END_ID = one("<|vision_end|>")
    PAD_TEXT_IDS.clear()
    for name, sid in (("<|image_pad|>", IMAGE_TOKEN_ID),
                      ("<|video_pad|>", VIDEO_TOKEN_ID)):
        if sid is None:
            continue
        try:
            plain = list(tok(name, add_special_tokens=False,
                             split_special_tokens=True)["input_ids"])
        except TypeError:
            plain = []
        if plain and sid not in plain and tok.decode(plain) == name:
            PAD_TEXT_IDS[sid] = plain
        else:
            print(f"serve_api: {name} has no text spelling on this "
                  f"tokenizer; a message that mentions it will be refused "
                  f"by the engine (issue #39)", flush=True)
    return IMAGE_TOKEN_ID

def textify_pad_mentions(ids, snap=0, snap2=0, n_images=0):
    if not PAD_TEXT_IDS or not any(v in PAD_TEXT_IDS for v in ids):
        return ids, snap, snap2, 0
    out, grown, n = [], 0, 0
    snap0, snap20, last = snap, snap2, len(ids) - 1
    for i, v in enumerate(ids):
        plain = PAD_TEXT_IDS.get(v)
        templ = (v == IMAGE_TOKEN_ID and n_images > 0 and 0 < i < last
                 and ids[i - 1] == VISION_START_ID
                 and ids[i + 1] == VISION_END_ID)
        if plain is not None and not templ:
            out.extend(plain)
            grown += len(plain) - 1
            n += 1
        else:
            out.append(v)
        if snap0 and i == snap0 - 1:
            snap = snap0 + grown
        if snap20 and i == snap20 - 1:
            snap2 = snap20 + grown
    return out, snap, snap2, n

def expand_image_pads(ids, snap, snap2, images):
    if not images:
        return ids, snap, snap2, []
    pads = [i for i, v in enumerate(ids) if v == IMAGE_TOKEN_ID]
    if len(pads) != len(images):
        raise HTTPException(400, f"the chat template emitted {len(pads)} "
                                 f"image placeholders for {len(images)} "
                                 f"images")
    out, placed, grown, k = [], [], 0, 0
    snap0, snap20 = snap, snap2
    for i, v in enumerate(ids):
        if k < len(pads) and i == pads[k]:
            H, W, _rgb, ntok = images[k]
            placed.append((len(out), H, W))
            out.extend([IMAGE_TOKEN_ID] * ntok)
            grown += ntok - 1
            k += 1
        else:
            out.append(v)
        if snap0 and i == snap0 - 1:
            snap = snap0 + grown
        if snap20 and i == snap20 - 1:
            snap2 = snap20 + grown
    return out, snap, snap2, placed

def vis_smart_resize(height, width, factor=VIS_UNIT,
                     min_pixels=VIS_MIN_PIXELS, max_pixels=VIS_MAX_PIXELS):
    if min(height, width) <= 0:
        raise ValueError("image has a zero dimension")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must be under 200:1")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def vis_decode_part(url):
    if not isinstance(url, str) or not url:
        raise ValueError("image_url.url must be a non-empty string")
    if url.startswith(("http://", "https://")):
        raise ValueError("this server does not fetch image URLs; inline the "
                         "image as a data: URL "
                         "(data:image/png;base64,...)")
    payload = url
    if url.startswith("data:"):
        head, _, payload = url.partition(",")
        if "base64" not in head:
            raise ValueError("only base64 data: URLs are supported")
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        raise ValueError("image_url.url is not valid base64")
    if not raw:
        raise ValueError("image payload is empty")
    try:
        from PIL import Image
    except ImportError:
        raise ValueError("this server was built without image support "
                         "(Pillow is not installed)")
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:
        raise ValueError(f"cannot decode the image: {e}")
    im = im.convert("RGB")
    W0, H0 = im.size
    if W0 * H0 > VIS_MAX_PIXELS * 4:
        raise ValueError(f"image is {W0}x{H0}; the limit before resizing is "
                         f"{VIS_MAX_PIXELS * 4} pixels")
    H, W = vis_smart_resize(H0, W0)
    if (W, H) != (W0, H0):
        from PIL import Image as _I
        im = im.resize((W, H), _I.BICUBIC)
    rgb = im.tobytes()
    if len(rgb) != H * W * 3:
        raise ValueError("decoded image is not 3-channel")
    ntok = (H // VIS_UNIT) * (W // VIS_UNIT)
    return H, W, rgb, ntok

def vis_collect(msgs):
    out = []
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for item in c:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url" or "image_url" in item:
                u = item.get("image_url")
                u = u.get("url") if isinstance(u, dict) else u
                out.append(vis_decode_part(u))
            elif item.get("type") == "image" or "image" in item:
                v = item.get("image")
                out.append(vis_decode_part(v if isinstance(v, str) else ""))
            elif item.get("type") == "video" or "video" in item:
                raise ValueError("this server has no video path; a `video` "
                                 "content part cannot be served")
    return out

def oai_error(status, message, code=None):
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": ERROR_TYPES.get(status, "server_error"),
        "param": None, "code": code}})

ANTH_ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
                    403: "permission_error", 404: "not_found_error",
                    413: "request_too_large", 429: "rate_limit_error",
                    503: "overloaded_error", 529: "overloaded_error"}
THINK_SIG = "hgn1."
BILLING_PREFIX = "x-anthropic-billing-header:"

def anth_error_obj(status, message):
    return {"type": ANTH_ERROR_TYPES.get(status, "api_error"),
            "message": message}

def anth_error(status, message):
    return JSONResponse(status_code=status, content={
        "type": "error", "error": anth_error_obj(status, message)})

def think_signature(text):
    return THINK_SIG + base64.urlsafe_b64encode(
        zlib.compress(text.encode("utf-8"))).decode("ascii")

def think_from_signature(sig):
    if not isinstance(sig, str) or not sig.startswith(THINK_SIG):
        return None
    try:
        return zlib.decompress(base64.urlsafe_b64decode(
            sig[len(THINK_SIG):].encode("ascii"))).decode("utf-8")
    except Exception:
        return None

def _anth_blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    raise ValueError("a message's `content` must be a string or a list of "
                     "content blocks")

def _anth_image(block, where):
    src = block.get("source") or {}
    if src.get("type") == "base64":
        return {"type": "image_url", "image_url": {
            "url": "data:%s;base64,%s" % (src.get("media_type") or "image/png",
                                          src.get("data") or "")}}
    if src.get("type") == "url" and src.get("url"):
        return {"type": "image_url", "image_url": {"url": src["url"]}}
    raise ValueError("%s: an image block needs `source` of type base64 or "
                     "url" % where)

def _anth_system_text(content, where):
    parts = []
    for b in _anth_blocks(content):
        if b.get("type") != "text":
            raise ValueError("%s: a system message may carry text only"
                             % where)
        t = b.get("text") or ""
        if t.startswith(BILLING_PREFIX):
            continue
        parts.append(t)
    return "\n\n".join(p for p in parts if p)

def _anth_tool_result(block, where):
    inner = block.get("content")
    parts = []
    for b in _anth_blocks(inner):
        t = b.get("type")
        if t == "text":
            parts.append({"type": "text", "text": b.get("text") or ""})
        elif t == "image":
            parts.append(_anth_image(b, where))
        else:
            raise ValueError("%s: a tool_result may hold text and images; "
                             "%r is not supported" % (where, t))
    if any(p["type"] == "image_url" for p in parts):
        return parts
    return "\n".join(p["text"] for p in parts)

def messages_to_openai(system, messages):
    out, head = [], []
    if system:
        head.append(_anth_system_text(system, "system"))
    seen_turn = False
    for i, m in enumerate(messages or []):
        if not isinstance(m, dict):
            raise ValueError("message %d is not an object" % i)
        role, where = m.get("role"), "message %d" % i
        if role == "system":
            text = _anth_system_text(m.get("content"), where)
            if not seen_turn:
                head.append(text)
            elif text:
                if out and out[-1]["role"] == "user":
                    out[-1]["content"].append({"type": "text", "text": text})
                else:
                    out.append({"role": "user",
                                "content": [{"type": "text", "text": text}]})
            continue
        seen_turn = True
        blocks = _anth_blocks(m.get("content"))
        if role == "user":
            parts = []
            for b in blocks:
                t = b.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": b.get("text") or ""})
                elif t == "image":
                    parts.append(_anth_image(b, where))
                elif t == "tool_result":
                    out.append({"role": "tool",
                                "tool_call_id": b.get("tool_use_id") or "",
                                "content": _anth_tool_result(b, where)})
                else:
                    raise ValueError("%s: content block type %r is not "
                                     "supported by this server" % (where, t))
            if parts:
                out.append({"role": "user", "content": parts})
        elif role == "assistant":
            reasoning, texts, calls = [], [], []
            for b in blocks:
                t = b.get("type")
                if t == "thinking":
                    r = think_from_signature(b.get("signature"))
                    reasoning.append(r if r is not None
                                     else (b.get("thinking") or ""))
                elif t == "redacted_thinking":
                    continue
                elif t == "text":
                    texts.append(b.get("text") or "")
                elif t == "tool_use":
                    calls.append({"id": b.get("id") or "", "type": "function",
                                  "function": {
                                      "name": b.get("name") or "",
                                      "arguments": json.dumps(
                                          b.get("input") or {},
                                          ensure_ascii=False)}})
                else:
                    raise ValueError("%s: content block type %r is not "
                                     "supported in an assistant turn"
                                     % (where, t))
            msg = {"role": "assistant", "content": "\n\n".join(texts)}
            reasoning = [r for r in reasoning if r]
            if reasoning:
                msg["reasoning_content"] = "\n\n".join(reasoning)
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            raise ValueError("%s has role %r; the Messages API takes user, "
                             "assistant and system" % (where, role))
    head = [h for h in head if h]
    if head:
        out.insert(0, {"role": "system", "content": "\n\n".join(head)})
    return out

CACHE_MARK = "\ue000halogen-cache-mark\ue000"

def mark_last_cache_point(messages, mark):
    last = None
    for i, m in enumerate(messages or []):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for j, b in enumerate(c):
                if isinstance(b, dict) and b.get("cache_control"):
                    last = (i, j)
    if last is None:
        return messages
    i, j = last
    c = messages[i]["content"]
    if i == len(messages) - 1 and j == len(c) - 1:
        return messages
    b = c[j]
    if b.get("type") == "text" and isinstance(b.get("text"), str):
        nb = dict(b, text=b["text"] + mark)
    elif b.get("type") == "tool_result" and isinstance(b.get("content"), str):
        nb = dict(b, content=b["content"] + mark)
    else:
        return messages
    out = list(messages)
    out[i] = dict(messages[i], content=c[:j] + [nb] + c[j + 1:])
    return out

def messages_tools(tools):
    out, dropped = [], []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        ty = t.get("type")
        if ty not in (None, "custom"):
            dropped.append(str(ty))
            continue
        out.append({"type": "function", "function": {
            "name": t.get("name") or "",
            "description": t.get("description") or "",
            "parameters": t.get("input_schema") or {"type": "object",
                                                   "properties": {}}}})
    return out or None, dropped

def messages_tool_choice(tc):
    if not isinstance(tc, dict):
        return None, None
    parallel = False if tc.get("disable_parallel_tool_use") else None
    ty = tc.get("type")
    if ty == "any":
        return "required", parallel
    if ty == "tool":
        return {"type": "function", "function": {"name": tc.get("name")}}, parallel
    if ty == "none":
        return "none", parallel
    return "auto", parallel

ENGINE_PING_S = float(os.environ.get("HALOGEN_ENGINE_PING_S", 30.0))
PROBE_S = float(os.environ.get("HALOGEN_ENGINE_PROBE_S", 5.0))
PROBE_ATTEMPTS = 3

FIRST_TOKEN_S = 1800.0
NEXT_TOKEN_S = 300.0
PREFIX_NOTE_MIN = 1024
ABORT_DRAIN_S = 10.0

KEEPALIVE_S = float(os.environ.get("HALOGEN_KEEPALIVE_TIMEOUT", 300.0))

SSE_KEEPALIVE_S = float(os.environ.get("HALOGEN_SSE_KEEPALIVE_S", 10.0))
PREFILL_RATE_MIN_TOKENS = 2048

def schema_text(rf):
    if rf.get("type") == "json_object":
        return "json"
    schema = rf.get("schema")
    if schema is None and isinstance(rf.get("json_schema"), dict):
        schema = rf["json_schema"].get("schema")
    if not isinstance(schema, dict):
        raise HTTPException(400, "response_format json_schema needs a "
                                 "`schema` object (OpenAI's `json_schema: "
                                 "{name, schema}` on chat, `text.format: "
                                 "{type, name, schema}` on /v1/responses)")
    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)

def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

_VOCAB_BLOB = None

def vocab_blob(tok):
    global _VOCAB_BLOB
    if _VOCAB_BLOB is not None:
        return _VOCAB_BLOB
    import struct
    u2b = {u: b for b, u in _bytes_to_unicode().items()}
    vocab = tok.get_vocab()
    added = set(tok.get_added_vocab().values()) if hasattr(tok, "get_added_vocab") else set()
    n = max(vocab.values()) + 1
    inv = {i: t for t, i in vocab.items()}
    out = bytearray(b"HGV1" + struct.pack("<I", n))
    for i in range(n):
        t = inv.get(i)
        if t is None:
            out += struct.pack("<BH", 1, 0)
        elif i in added:
            b = t.encode("utf-8")
            out += struct.pack("<BH", 1, len(b)) + b
        else:
            b = bytes(u2b[ch] for ch in t)
            out += struct.pack("<BH", 0, len(b)) + b
    _VOCAB_BLOB = (n, bytes(out))
    return _VOCAB_BLOB

def t_line_logprob(parts):
    if len(parts) <= 3:
        return None
    lp = float(parts[3])
    if len(parts) > 4:
        n = int(parts[4])
        lp = (lp, [(int(parts[5 + 2 * j]), float(parts[6 + 2 * j]))
                   for j in range(n) if int(parts[5 + 2 * j]) >= 0])
    return lp

_TOKEN_BYTES = None

def token_bytes(tok, i):
    global _TOKEN_BYTES
    if _TOKEN_BYTES is None:
        u2b = {u: b for b, u in _bytes_to_unicode().items()}
        added = set(tok.get_added_vocab().values()) if hasattr(tok, "get_added_vocab") else set()
        table = {}
        for t, j in tok.get_vocab().items():
            if j in added:
                table[j] = t.encode("utf-8")
            else:
                try:
                    table[j] = bytes(u2b[ch] for ch in t)
                except KeyError:
                    table[j] = t.encode("utf-8")
        _TOKEN_BYTES = table
    return _TOKEN_BYTES.get(int(i), b"")

def logprob_entry(tok, tid, v):
    lp, tops = v if isinstance(v, tuple) else (v, None)

    def one(i, x):
        b = token_bytes(tok, i)
        return {"token": b.decode("utf-8", "replace"), "logprob": x,
                "bytes": list(b)}
    e = one(tid, lp)
    e["top_logprobs"] = [one(i, x) for i, x in (tops or [])]
    return e

PRIORITY_HEADER = "X-Halogen-Priority"

def priority_hinted(http_request):
    try:
        v = http_request.headers.get(PRIORITY_HEADER, "")
    except AttributeError:
        return False
    return v.strip().lower() in ("1", "true", "high", "yes")

class Metrics:
    def __init__(self):
        self.prompt_tokens = 0
        self.prompt_seconds = 0.0
        self.predicted_tokens = 0
        self.predicted_seconds = 0.0
        self.requests = 0
        self.cache_tokens = 0
        self.draft_tokens = 0
        self.draft_accepted = 0
        self.structured_requests = 0
        self.dv_tokens = 0
        self.dv_misses = 0
        self.b_prompt_tokens = 0
        self.b_prompt_seconds = 0.0
        self.b_predicted_tokens = 0
        self.b_predicted_seconds = 0.0

    def record(self, t, structured=False, d=None):
        if d:
            self.dv_tokens += int(d.get("dv_tokens", 0))
            self.dv_misses += int(d.get("dv_misses", 0))
        pn = int(t.get("prompt_n", 0)); gn = int(t.get("predicted_n", 0))
        ps = float(t.get("prompt_ms", 0.0)) / 1000.0
        gs = float(t.get("predicted_ms", 0.0)) / 1000.0
        self.prompt_tokens += pn; self.prompt_seconds += ps
        self.predicted_tokens += gn; self.predicted_seconds += gs
        self.b_prompt_tokens += pn; self.b_prompt_seconds += ps
        self.b_predicted_tokens += gn; self.b_predicted_seconds += gs
        self.requests += 1
        self.cache_tokens += int(t.get("cache_n", 0))
        self.draft_tokens += int(t.get("draft_n", 0))
        self.draft_accepted += int(t.get("draft_n_accepted", 0))
        self.structured_requests += 1 if structured else 0

    def render(self, processing, deferred, kv_reserved, kv_pool, kv_used=0, admission=None):
        b_ps = (self.b_prompt_tokens / self.b_prompt_seconds
                if self.b_prompt_seconds > 0 and self.b_prompt_tokens > 0 else 0.0)
        b_gs = (self.b_predicted_tokens / self.b_predicted_seconds
                if self.b_predicted_seconds > 0 else 0.0)
        self.b_prompt_tokens = 0; self.b_prompt_seconds = 0.0
        self.b_predicted_tokens = 0; self.b_predicted_seconds = 0.0
        rows = [
            ("counter", "llamacpp:prompt_tokens_total", "Number of prompt tokens processed.", self.prompt_tokens),
            ("counter", "llamacpp:prompt_seconds_total", "Prompt process time.", self.prompt_seconds),
            ("counter", "llamacpp:tokens_predicted_total", "Number of generation tokens processed.", self.predicted_tokens),
            ("counter", "llamacpp:tokens_predicted_seconds_total", "Predict process time.", self.predicted_seconds),
            ("gauge", "llamacpp:prompt_tokens_seconds", "Average prompt throughput in tokens/s. (since the last scrape)", b_ps),
            ("gauge", "llamacpp:predicted_tokens_seconds", "Average generation throughput in tokens/s. (since the last scrape)", b_gs),
            ("gauge", "llamacpp:requests_processing", "Number of requests processing.", processing),
            ("gauge", "llamacpp:requests_deferred", "Number of requests deferred.", deferred),
        ]
        if kv_pool:
            rows += [
                ("gauge", "llamacpp:kv_cache_tokens", "KV-cache tokens (positions the engine's pool holds, busy and held regions).", kv_used),
                ("gauge", "llamacpp:kv_cache_usage_ratio", "KV-cache usage. 1 means 100 percent usage. (positions held over the pool)", min(1.0, kv_used / kv_pool)),
            ]
        rows += [
            ("counter", "halogen:requests_total", "Requests completed.", self.requests),
            ("counter", "halogen:prompt_tokens_cached_total", "Prompt tokens the prompt cache covered (not processed).", self.cache_tokens),
            ("counter", "halogen:draft_tokens_total", "Tokens proposed by the draft head and prompt lookup.", self.draft_tokens),
            ("counter", "halogen:draft_tokens_accepted_total", "Of those, accepted.", self.draft_accepted),
            ("counter", "halogen:structured_requests_total", "Requests decoded under a JSON schema.", self.structured_requests),
            ("counter", "halogen:draft_list_tokens_total", "Tokens the draft head drafted on a short list.", self.dv_tokens),
            ("counter", "halogen:draft_list_misses_total", "Of those, tokens outside the list.", self.dv_misses),
        ]
        if kv_pool:
            rows += [
                ("gauge", "halogen:kv_pool_positions", "The KV pool, in positions.", kv_pool),
                ("gauge", "halogen:kv_pool_reserved_tokens", "Positions the requests holding a front-end slot asked for (prompt + max_tokens each), admitted or not.", kv_reserved),
            ]
        if admission:
            rows += [
                ("gauge", "halogen:admission_reserve_slots", "Engine slots held back from requests without the priority header.", admission[0]),
                ("gauge", "halogen:admission_priority_in_flight", "Requests in flight that sent the priority header.", admission[1]),
                ("gauge", "halogen:admission_waiting_for_reserve", "Requests waiting only because the reserve holds the free slots back.", admission[2]),
            ]
        out = []
        for kind, name, help_, value in rows:
            out.append("# HELP %s %s" % (name, help_))
            out.append("# TYPE %s %s" % (name, kind))
            out.append("%s %s" % (name, ("%d" % value) if isinstance(value, int) else ("%.6g" % value)))
        return "\n".join(out) + "\n"

class EngineBusy(HTTPException):
    def __init__(self):
        super().__init__(status_code=503,
                         detail="timed out waiting for the engine: "
                                "halogen serves one request at a time "
                                "(batch-1) and the queue did not clear")

def engine_refusal(parts):
    tail = parts[7:]
    i = 0
    while i < len(tail):
        try:
            float(tail[i])
        except ValueError:
            break
        i += 1
    why = " ".join(tail[i:]).strip()
    if why:
        return "the engine refused this request: " + why
    return ("the engine rejected this request. Check /health for what this "
            "build supports, and that the prompt fits the context.")

class Engine:

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.r = self.w = None
        self.slots = asyncio.Semaphore(1)
        self.n_slots = 1
        try:
            self.reserve = max(0, int(os.environ.get("HALOGEN_ADMISSION_RESERVE", "0") or 0))
        except ValueError:
            self.reserve = 0
        self.bg = None
        self.bg_of = {}
        self.waiting_reserve = 0
        self.priority_inflight = 0
        self.hinted_keys = set()
        self.req = 0
        self.active = None
        self.waiting = 0
        self.inflight = {}
        self.busy_since = None
        self.streams = {}
        self.wlock = asyncio.Lock()
        self.clock = asyncio.Lock()
        self.reader = None
        self.cstat = asyncio.Queue()
        self.info = {}
        self.fallback_ctx = 0
        self.tokenizer = None
        self.metrics = Metrics()
        self.reserved = {}
        self.pool_used = 0
        self.pool_positions = 0
        self._live = (True, 0.0, "")
        self._live_at = None
        self._cstat_last = None
        self._cstat_at = None

    async def connect(self):
        async with self.clock:
            await self._connect_locked()

    async def _connect_locked(self):
        if self.w is not None and not self.w.is_closing():
            return
        for q in list(self.streams.values()):
            q.put_nowait(None)
        self.streams = {}
        if self.reader is not None and not self.reader.done():
            self.reader.cancel()
        self.reader = None
        old_w = self.w
        self.r, self.w = await asyncio.open_connection(self.host, self.port)
        if old_w is not None:
            try:
                old_w.close()
            except Exception:
                pass
        self.info = await self._probe()
        await self._apply_info()

    async def _apply_info(self):
        if self.info.get("grammar"):
            await self._send_vocab()
        n = max(1, int(self.info.get("kv_slots", 1)))
        if n != self.n_slots or self.slots is None:
            self.n_slots = n
            self.slots = asyncio.Semaphore(n)
            self.bg = None
        if self.reserve and self.bg is None:
            keep = min(self.reserve, self.n_slots - 1)
            if keep < self.reserve:
                print(f"serve_api: HALOGEN_ADMISSION_RESERVE {self.reserve} is "
                      f"more than {self.n_slots} slot(s) can spare: holding "
                      f"back {keep}", flush=True)
            if keep > 0:
                self.bg = asyncio.Semaphore(self.n_slots - keep)
                print(f"serve_api: admission reserve on: {keep} of "
                      f"{self.n_slots} slot(s) held for requests sending "
                      f"{PRIORITY_HEADER}", flush=True)
        if self.n_slots > 1:
            self.reader = asyncio.create_task(self._demux(self.streams))

    async def admit(self, hinted, timeout):
        bg = None if hinted else self.bg
        self.waiting += 1
        t_end = time.monotonic() + timeout
        try:
            if bg is not None:
                self.waiting_reserve += 1
                try:
                    await asyncio.wait_for(bg.acquire(), timeout=timeout)
                finally:
                    self.waiting_reserve -= 1
            try:
                await asyncio.wait_for(self.slots.acquire(),
                                       timeout=max(0.0, t_end - time.monotonic()))
            except BaseException:
                if bg is not None:
                    bg.release()
                raise
        except asyncio.TimeoutError:
            raise EngineBusy()
        finally:
            self.waiting -= 1
        return bg

    def release_admission(self, slot_key):
        bg = self.bg_of.pop(slot_key, None)
        if bg is not None:
            bg.release()
        if slot_key in self.hinted_keys:
            self.hinted_keys.discard(slot_key)
            self.priority_inflight -= 1

    async def _send_vocab(self):
        try:
            if self.tokenizer is None:
                raise RuntimeError("no tokenizer bound to the engine client")
            blob = vocab_blob(self.tokenizer)
            self.w.write(b"VOCAB %d %s\n" % (blob[0], base64.b64encode(blob[1])))
            await self.w.drain()
            for _ in range(PROBE_ATTEMPTS + 1):
                raw = await asyncio.wait_for(self.r.readline(), timeout=60)
                p = raw.decode().split(None, 2)
                if not raw or not p or p[0] != "I":
                    break
            if len(p) >= 2 and p[0] == "V" and p[1] == "ok":
                return
            self.info["grammar"] = False
            self.info["grammar_error"] = (" ".join(p[2:]) or raw.decode().strip()
                                          or "no reply to VOCAB")
        except Exception as e:
            self.info["grammar"] = False
            self.info["grammar_error"] = f"VOCAB failed: {e}"
        print(f"serve_api: structured output OFF: {self.info['grammar_error']}",
              flush=True)

    async def probe_live(self, timeout=None):
        now = time.monotonic()
        if self._live_at is not None and now - self._live_at < 1.0:
            return self._live
        t = float(timeout if timeout is not None else ENGINE_PING_S)
        t0 = time.monotonic()
        w = None

        async def ask():
            nonlocal w
            r, w = await asyncio.open_connection(self.host, self.port)
            w.write(b"PING\n")
            await w.drain()
            return await r.readline()

        try:
            line = await asyncio.wait_for(ask(), timeout=t)
            ok = line.strip() == b"PONG"
            out = (ok, round(time.monotonic() - t0, 3),
                   "" if ok else f"unexpected reply {line[:40]!r}")
        except asyncio.TimeoutError:
            out = (False, round(time.monotonic() - t0, 3),
                   f"no PONG within {t:g}s")
        except Exception as e:
            out = (False, round(time.monotonic() - t0, 3), repr(e)[:120])
        finally:
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
        self._live, self._live_at = out, time.monotonic()
        return out

    async def _demux(self, streams=None):
        try:
            while True:
                raw = await self.r.readline()
                if not raw:
                    break
                parts = raw.decode().split()
                if parts and parts[0] == "C":
                    self.cstat.put_nowait(parts)
                    continue
                if len(parts) < 2 or parts[0] not in ("T", "D"):
                    continue
                try:
                    rid = int(parts[1])
                except ValueError:
                    continue
                q = self.streams.get(rid)
                if q is not None:
                    q.put_nowait(parts)
        except Exception:
            pass
        finally:
            for q in list((streams if streams is not None else self.streams).values()):
                q.put_nowait(None)

    async def _probe(self):
        for attempt in range(PROBE_ATTEMPTS):
            try:
                self.w.write(b"INFO\n")
                await self.w.drain()
                raw = await asyncio.wait_for(self.r.readline(), timeout=PROBE_S)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"serve_api: capability probe FAILED: {e}", flush=True)
                break
            if not raw:
                break
            p = raw.decode().split()
            if p and p[0] == "I" and len(p) >= 6:
                try:
                    return self._parse_info(p)
                except Exception as e:
                    print(f"serve_api: capability probe FAILED to parse "
                          f"{raw.decode().strip()!r}: {e}", flush=True)
                    break
        print(f"serve_api: capability probe FAILED: the engine did not answer "
              f"INFO in {PROBE_ATTEMPTS} x {PROBE_S:g} s. Serving on defaults "
              f"(context {self.fallback_ctx}, one slot, no sampling, no "
              f"images, no grammar) until a request re-probes it; /health "
              f"reports capability_probe=failed meanwhile.", flush=True)
        return self._fallback_info()

    @staticmethod
    def _fallback_info():
        return {"mtp": False, "draft_head": False, "default": 0,
                "kv_slots": 1, "slot_ctx": 0, "cache_mode": 1,
                "sampling": False, "rope_factor": 1.0, "kv_pool": 0,
                "vision": False, "version": "unknown", "probed": False}

    async def _reprobe(self):
        if self.info.get("probed") or self.reader is not None \
                or self.active is not None:
            return
        if self.w is None or self.w.is_closing():
            return
        try:
            self.w.write(b"INFO\n")
            await self.w.drain()
            raw = await asyncio.wait_for(self.r.readline(), timeout=PROBE_S)
            p = raw.decode().split()
            if p and p[0] == "I" and len(p) >= 6:
                self.info = self._parse_info(p)
        except Exception:
            return
        if self.info.get("probed"):
            print(f"serve_api: capability probe recovered: context "
                  f"{self.info['ctx']}, {self.info['kv_slots']} slot(s), "
                  f"engine version {self.info.get('version')}", flush=True)
            await self._apply_info()

    @staticmethod
    def _parse_info(p):
        return {"probed": True,
                "mtp": p[1] == "1", "draft_head": p[2] == "1",
                "ctx": int(p[3]), "spec_rows": int(p[4]),
                "default": int(p[5]),
                "drafter_weights": len(p) >= 7 and p[6] == "1",
                "dflash2": len(p) >= 8 and p[7] == "1",
                "cache_mb": int(p[8]) if len(p) >= 9 else 0,
                "cache_align": int(p[9]) if len(p) >= 10 else 0,
                "kv_slots": int(p[10]) if len(p) >= 11 else 1,
                "slot_ctx": int(p[11]) if len(p) >= 12 else 0,
                "cache_mode": int(p[12]) if len(p) >= 13 else 1,
                "sampling": len(p) >= 14 and p[13] == "1",
                "rope_factor": float(p[14]) if len(p) >= 15 else 1.0,
                "kv_pool": int(p[15]) if len(p) >= 16 else 0,
                "vision": len(p) >= 17 and p[16] == "1",
                "version": p[17] if len(p) >= 18 else "unknown",
                "pld_n": int(p[18]) if len(p) >= 20 else 0,
                "pld_k": int(p[19]) if len(p) >= 20 else 0,
                "checkpoint_format": p[20] if len(p) >= 21 else "hgn",
                "grammar": len(p) >= 22 and p[21] == "1",
                "composable": len(p) >= 23 and p[22] == "1",
                "cc_floor": int(p[23]) if len(p) >= 24 else 0,
                "cc_seam": int(p[24]) if len(p) >= 25 else 0,
                "cc_bytes": int(p[25]) if len(p) >= 26 else 0,
                "cc_fingerprint": p[26] if len(p) >= 27 else "-",
                "indexer_budget": int(p[27]) if len(p) >= 28 else 2048}

    async def _ensure(self):
        if self.w is None or self.w.is_closing():
            await self.connect()
        elif not self.info.get("probed"):
            async with self.clock:
                await self._reprobe()

    async def close(self):
        try:
            if self.w is not None:
                self.w.close()
        except Exception:
            pass
        self.r = self.w = None
        self.active = None

    @staticmethod
    def _done_dict(parts):
        d = {"reason": parts[2], "n_prompt": int(parts[3]),
             "n_gen": int(parts[4]), "prefill_ms": float(parts[5]),
             "decode_ms": float(parts[6])}
        if len(parts) >= 10:
            d["drafter"] = int(parts[7])
            d["rounds"] = int(parts[8])
            d["commit"] = int(parts[9])
        if len(parts) >= 11:
            d["n_cached"] = int(parts[10])
        if len(parts) >= 12:
            d["shared"] = int(parts[11])
        if len(parts) >= 14:
            d["pld_rounds"] = int(parts[12])
            d["pld_accepted"] = int(parts[13])
        if len(parts) >= 15:
            d["pld_drafted"] = int(parts[14])
        if len(parts) >= 16:
            d["n_composed"] = int(parts[15])
        if len(parts) >= 20:
            d["room_from"] = int(parts[16])
            d["max_tokens"] = int(parts[17])
            d["pool_used"] = int(parts[18])
            d["pool_positions"] = int(parts[19])
        if len(parts) >= 22:
            d["think_forced"] = int(parts[20])
            d["think_seen"] = int(parts[21])
        if len(parts) >= 24:
            d["kept_think"] = int(parts[22])
            d["kept_call"] = int(parts[23])
        if len(parts) >= 26:
            d["disk_rows"] = int(parts[24])
            d["disk_ms"] = float(parts[25])
        if len(parts) >= 27:
            d["head_drafted"] = int(parts[26])
        if len(parts) >= 30:
            d["dv_tokens"] = int(parts[27])
            d["dv_misses"] = int(parts[28])
            d["dv_switches"] = int(parts[29])
        return d

    def _note_pool(self, d):
        if d.get("pool_positions"):
            self.pool_used = int(d["pool_used"])
            self.pool_positions = int(d["pool_positions"])

    async def _gen_batched(self, req, line, ctl=None):
        q = asyncio.Queue()
        self.streams[req] = q
        finished = False
        cancel_at = None
        try:
            async with self.wlock:
                self.w.write(line.encode())
                await self.w.drain()
            first = True
            while True:
                if ctl is not None and ctl.get("stop") and cancel_at is None:
                    cancel_at = time.monotonic() + ABORT_DRAIN_S
                    try:
                        async with self.wlock:
                            self.w.write(f"X {req}\n".encode())
                            await self.w.drain()
                    except Exception:
                        return
                timeout = FIRST_TOKEN_S if first else NEXT_TOKEN_S
                if cancel_at is not None:
                    timeout = cancel_at - time.monotonic()
                    if timeout <= 0:
                        return
                try:
                    parts = await asyncio.wait_for(q.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    if cancel_at is not None:
                        return
                    raise HTTPException(
                        504, "engine went silent for %ds%s"
                             % (FIRST_TOKEN_S if first else NEXT_TOKEN_S,
                                " while prefilling" if first else " mid-decode"))
                if parts is None:
                    if cancel_at is not None:
                        return
                    raise HTTPException(502, "engine closed the connection")
                if parts[0] == "T":
                    first = False
                    if cancel_at is not None:
                        continue
                    yield (int(parts[2]), None, t_line_logprob(parts))
                elif parts[0] == "D":
                    if parts[2] == "error":
                        if cancel_at is not None:
                            return
                        raise HTTPException(400, engine_refusal(parts))
                    finished = True
                    d = self._done_dict(parts)
                    self._note_pool(d)
                    yield None, d, None
                    return
        finally:
            self.streams.pop(req, None)
            if not finished and cancel_at is None:
                try:
                    self.w.write(f"X {req}\n".encode())
                except Exception:
                    pass

    async def generate(self, ids, max_tokens, eos, drafter=None, sample=None,
                       penalty="", snap=0, snap2=0, images=None, schema=None,
                       after=None, escape=(), think=None, seg=None,
                       snap3=0, guard=None, ctl=None):
        await self._ensure()
        self.req += 1
        req = self.req
        for first_tok, H, W, rgb in (images or []):
            self.w.write(("VIMG %d %d %d %d %s\n" %
                          (req, first_tok, H, W,
                           base64.b64encode(rgb).decode())).encode())
        line = (f"GEN {req} {max_tokens} {len(eos)} "
                + (" ".join(str(e) for e in eos) + " " if eos else "")
                + f"{len(ids)} " + " ".join(str(i) for i in ids)
                + (f" {drafter}" if drafter is not None else "")
                + (" SAMPLE %.9g %d %.9g %.9g %d" % sample if sample else "")
                + (penalty or "")
                + (f" SNAP {snap}" if snap else "")
                + (f" SNAP2 {snap2}" if snap2 else "")
                + (f" SNAP3 {snap3}" if snap3 else "")
                + ((" IMG %d " % len(images)) +
                   " ".join("%d %d %d" % (f, h, w) for f, h, w, _ in images)
                   if images else "")
                + ((" SCHEMA json" if schema == "json" else
                    " SCHEMA " + base64.b64encode(schema.encode()).decode())
                   if schema else "")
                + (f" AFTER {after}" if schema and after is not None else "")
                + ((" ESCAPE %d " % len(escape)) + " ".join(str(e) for e in escape)
                   if schema and escape else "")
                + ((" THINK %d %d %d " % (think[0], think[1], len(think[2])))
                   + " ".join(str(t) for t in think[2]) if think else "")
                + (" GUARD %d %d %d %d %d" % guard if guard else "")
                + ((" SEG %d " % len(seg)) + " ".join(str(x) for x in seg)
                   if seg and self.info.get("composable") else "")
                + "\n")
        if self.n_slots > 1:
            async for item in self._gen_batched(req, line, ctl):
                yield item
            return
        self.w.write(line.encode())
        await self.w.drain()
        self.active = req
        first = True
        cancel_at = None
        while True:
            if ctl is not None and ctl.get("stop") and cancel_at is None:
                cancel_at = time.monotonic() + ABORT_DRAIN_S
                try:
                    self.w.write(f"X {req}\n".encode())
                    await self.w.drain()
                except Exception:
                    await self.close()
                    return
            timeout = FIRST_TOKEN_S if first else NEXT_TOKEN_S
            if cancel_at is not None:
                timeout = cancel_at - time.monotonic()
                if timeout <= 0:
                    await self.close()
                    return
            try:
                raw = await asyncio.wait_for(self.r.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                if cancel_at is not None:
                    await self.close()
                    return
                await self.close()
                raise HTTPException(
                    504, "engine went silent for %ds%s. The connection was "
                         "dropped and the next request will reconnect"
                         % (FIRST_TOKEN_S if first else NEXT_TOKEN_S,
                            " while prefilling" if first else " mid-decode"))
            if not raw:
                self.active = None
                if cancel_at is not None:
                    return
                raise HTTPException(502, "engine closed the connection")
            parts = raw.decode().split()
            if not parts:
                continue
            if parts[0] == "T" and int(parts[1]) == req:
                first = False
                if cancel_at is not None:
                    continue
                yield (int(parts[2]), None, t_line_logprob(parts))
            elif parts[0] == "D" and int(parts[1]) == req:
                self.active = None
                if parts[2] == "error" and cancel_at is not None:
                    return
                if parts[2] == "error":
                    raise HTTPException(400, engine_refusal(parts))
                d = self._done_dict(parts)
                self._note_pool(d)
                yield None, d, None
                return

    async def cache_stats(self):
        await self._ensure()
        if self.n_slots > 1:
            while not self.cstat.empty():
                self.cstat.get_nowait()
            async with self.wlock:
                self.w.write(b"CSTAT\n")
                await self.w.drain()
            try:
                p = await asyncio.wait_for(self.cstat.get(), timeout=10)
            except asyncio.TimeoutError:
                if self._cstat_last is None:
                    raise HTTPException(503, "the engine is inside a long request and has not "
                                        "reported its cache counters yet; try again in a moment")
                d = dict(self._cstat_last)
                d["stale_s"] = round(time.monotonic() - self._cstat_at, 1)
                return d
        else:
            async with self.slots:
                self.w.write(b"CSTAT\n")
                await self.w.drain()
                raw = await asyncio.wait_for(self.r.readline(), timeout=10)
            p = raw.decode().split()
        if not p or p[0] != "C" or len(p) < 10:
            return None
        d = {"entries": int(p[1]), "bytes": int(p[2]),
             "last_entry_bytes": int(p[3]), "max_entries": int(p[4]),
             "hits": int(p[5]), "misses": int(p[6]), "stores": int(p[7]),
             "evicted": int(p[8]), "prompt_tokens_saved": int(p[9])}
        if len(p) >= 14:
            d.update(store_ms_total=float(p[10]),
                     restore_ms_total=float(p[11]),
                     last_store_ms=float(p[12]),
                     last_restore_ms=float(p[13]))
        if len(p) >= 15:
            d["refused"] = int(p[14])
        if len(p) >= 17:
            d["rows_copied"] = int(p[15])
            d["rows_copied_ms"] = float(p[16])
        if len(p) >= 18:
            d["superseded"] = int(p[17])
        if len(p) >= 24:
            d["composable_context"] = {
                "chunks": int(p[18]), "bytes": int(p[19]),
                "stored": int(p[20]), "composed": int(p[21]),
                "evicted": int(p[22]), "composed_tokens": int(p[23])}
        if len(p) >= 37:
            d["disk"] = {
                "on": p[24] == "1", "records": int(p[25]), "lineages": int(p[26]),
                "bytes": int(p[27]), "budget_bytes": int(p[28]),
                "hits": int(p[29]), "misses": int(p[30]), "persisted": int(p[31]),
                "branches": int(p[32]), "evicted": int(p[33]), "skipped": int(p[34]),
                "restore_ms_total": float(p[35]), "restored_bytes": int(p[36])}
        if len(p) >= 39:
            d["tapped"] = int(p[37])
            d["full_hits"] = int(p[38])
        if len(p) >= 43:
            d["pool"] = {"waiting_for_room": int(p[39]),
                         "waiting_s": float(p[40]),
                         "relocated": int(p[41]),
                         "cold_resorts": int(p[42])}
        if len(p) >= 49:
            d["pool"].update(positions=int(p[44]), used=int(p[43]),
                             busy_regions=int(p[45]), held_regions=int(p[46]),
                             room_clamped=int(p[47]), moved=int(p[48]))
            self._note_pool({"pool_used": p[43], "pool_positions": p[44]})
        if len(p) >= 50:
            d["pool"]["packed"] = int(p[49])
        if len(p) >= 51:
            d["dropped"] = int(p[50])
        if len(p) >= 52:
            d["pool"]["taken_over"] = int(p[51])
        self._cstat_last = dict(d)
        self._cstat_at = time.monotonic()
        return d

    async def abort(self):
        if self.n_slots > 1:
            return
        req = self.active
        if req is None:
            return
        try:
            if self.w is not None and not self.w.is_closing():
                self.w.write(f"X {req}\n".encode())
                await self.w.drain()
            deadline = time.monotonic() + ABORT_DRAIN_S
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                raw = await asyncio.wait_for(self.r.readline(), timeout=left)
                if not raw:
                    break
                p = raw.decode().split()
                if p and p[0] == "D" and int(p[1]) == req:
                    self.active = None
                    return
        except Exception:
            pass
        await self.close()

EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium",
              "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
QWEN_BUDGET_SENTENCE = ("Considering the limited time by the user, I have to "
                        "give the solution based on the thinking directly now.")
THINK_CLOSE_MAX = 512
EFFORT_NONE = "none"
EFFORT_VALUES = list(EFFORT_MAP) + [EFFORT_NONE]

def effort_is_none(req):
    e = getattr(req, "reasoning_effort", None)
    return isinstance(e, str) and e.strip().lower() == EFFORT_NONE

def thinking_on(req):
    if effort_is_none(req):
        return False
    return server_default(req, "enable_thinking") is not False

def _env_number(name, lo, hi, kind):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        v = kind(raw)
    except ValueError:
        raise SystemExit(f"serve_api: {name}={raw!r} is not a "
                         f"{'whole number' if kind is int else 'number'}")
    if not (lo <= v <= hi):
        raise SystemExit(f"serve_api: {name}={raw} is outside {lo}..{hi}; "
                         f"values are not clamped, fix the variable")
    return v

def _env_effort(name):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    e = EFFORT_MAP.get(raw.strip().lower())
    if e is None:
        if raw.strip().lower() == EFFORT_NONE:
            raise SystemExit(f"serve_api: {name}=none: the default for "
                             f"thinking OFF is HALOGEN_ENABLE_THINKING=0; "
                             f"{name} takes {'|'.join(EFFORT_MAP)}")
        raise SystemExit(f"serve_api: {name}={raw!r} unsupported; use "
                         f"{'|'.join(EFFORT_MAP)}")
    return e

def _env_bool(name):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"serve_api: {name}={raw!r} is not a boolean; use 1 or 0")

DEFAULTS = {
    "max_tokens": _env_number("HALOGEN_MAX_TOKENS_DEFAULT", 1, 1 << 30, int),
    "temperature": _env_number("HALOGEN_TEMPERATURE", 0.0, 2.0, float),
    "top_p": _env_number("HALOGEN_TOP_P", 0.0, 1.0, float),
    "top_k": _env_number("HALOGEN_TOP_K", 0, 1 << 30, int),
    "min_p": _env_number("HALOGEN_MIN_P", 0.0, 1.0, float),
    "presence_penalty": _env_number("HALOGEN_PRESENCE_PENALTY", -2.0, 2.0,
                                    float),
    "frequency_penalty": _env_number("HALOGEN_FREQUENCY_PENALTY", -2.0, 2.0,
                                     float),
    "repetition_penalty": _env_number("HALOGEN_REPETITION_PENALTY", 0.0,
                                      float("inf"), float),
    "reasoning_effort": _env_effort("HALOGEN_REASONING_EFFORT"),
    "enable_thinking": _env_bool("HALOGEN_ENABLE_THINKING"),
    "max_thinking_tokens": _env_number("HALOGEN_MAX_THINKING_TOKENS", 1, 1 << 30, int),
    "thinking_answer_room": _env_number("HALOGEN_THINKING_ANSWER_ROOM", 0, 1 << 30, int),
}

def answer_room(max_tokens):
    r = DEFAULTS["thinking_answer_room"]
    if r is None:
        return max(1024, int(max_tokens) * 15 // 100)
    return int(r)

EOS_GUARD = _env_bool("HALOGEN_EOS_GUARD") is not False

TEMPLATE_UNCHECKED = _env_bool("HALOGEN_TEMPLATE_UNCHECKED") is True
THINK_EMPTY_BLOCK = "<think>\n\n</think>"

def probe_template(tok, where):
    import hashlib
    text = getattr(tok, "chat_template", None)
    if not isinstance(text, str) or not text:
        return {"path": str(where), "sha256": "", "probe": "failed",
                "thinking_control": False,
                "why": f"the tokenizer at {where} carries no chat template"}
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = f"{where} (embedded)"
    if os.path.isdir(where):
        for cand in ("chat_template.jinja", "tokenizer_config.json"):
            f = os.path.join(where, cand)
            if not os.path.isfile(f):
                continue
            try:
                raw = open(f, encoding="utf-8").read()
                if cand.endswith(".json"):
                    raw = json.loads(raw).get("chat_template")
            except (OSError, ValueError):
                continue
            if raw == text:
                path = f
                break
    msgs = [{"role": "user", "content": "hi"}]

    def render(on):
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=on)
    try:
        off, on = render(False), render(True)
    except Exception as e:
        return {"path": path, "sha256": sha, "probe": "failed",
                "thinking_control": False,
                "why": f"the chat template at {path} failed to render with "
                       f"enable_thinking ({type(e).__name__}: {e})"}
    why = None
    if off == on:
        why = (f"the chat template at {path} renders the same prompt with "
               f"enable_thinking false and true, so it has no thinking "
               f"control: enable_thinking, reasoning_effort \"none\" and "
               f"HALOGEN_ENABLE_THINKING=0 would be accepted and ignored")
    elif THINK_EMPTY_BLOCK not in off:
        why = (f"the chat template at {path} does not write the empty think "
               f"block ({THINK_EMPTY_BLOCK!r}) with enable_thinking false, "
               f"so a thinking-off reply would be misfiled as reasoning")
    elif "</think>" in on:
        why = (f"the chat template at {path} closes the think block in the "
               f"prompt with enable_thinking true, so the model would never "
               f"think and the reply would be misfiled as content")
    return {"path": path, "sha256": sha,
            "probe": "passed" if why is None else "failed",
            "thinking_control": why is None, "why": why}

def template_line(info):
    return (f"serve_api: chat template {info['path']} sha256 "
            f"{info['sha256'][:16]}, thinking control probe "
            f"{'passed' if info['probe'] == 'passed' else 'FAILED'}"
            + (f" (HALOGEN_TEMPLATE_UNCHECKED=1, serving anyway)"
               if info["probe"] != "passed" and TEMPLATE_UNCHECKED else ""))

def check_template_or_exit(tok, where):
    info = probe_template(tok, where)
    if info["probe"] != "passed":
        print(f"serve_api: {info['why']}", flush=True)
        if not TEMPLATE_UNCHECKED:
            print("serve_api: refusing to start on this chat template; mount "
                  "the tokenizer directory from the weights repo, or set "
                  "HALOGEN_TEMPLATE_UNCHECKED=1 to serve it anyway",
                  flush=True)
            print(template_line(info), flush=True)
            raise SystemExit(1)
        info["probe"] = "unchecked"
    print(template_line(info), flush=True)
    return info
CHAT_MAX_TOKENS = DEFAULTS["max_tokens"] or 8192
COMPLETION_MAX_TOKENS = DEFAULTS["max_tokens"] or 128
SAMPLING_DEFAULT_FIELDS = ("temperature", "top_p", "top_k", "min_p",
                           "presence_penalty", "frequency_penalty",
                           "repetition_penalty")
if DEFAULTS["repetition_penalty"] is not None and not DEFAULTS["repetition_penalty"] > 0:
    raise SystemExit("serve_api: HALOGEN_REPETITION_PENALTY must be greater "
                     "than 0 (1 = off)")

def server_default(req, field):
    v = getattr(req, field, None)
    return v if v is not None else DEFAULTS.get(field)

MODEL_FILTERS_BUILTIN = {"top_k": 20, "top_p": 0.95}
MODEL_FILTERS = dict(MODEL_FILTERS_BUILTIN)
MODEL_FILTERS_SOURCE = "built-in"

def load_model_filters(tok_dir):
    global MODEL_FILTERS_SOURCE
    MODEL_FILTERS.clear()
    MODEL_FILTERS.update(MODEL_FILTERS_BUILTIN)
    MODEL_FILTERS_SOURCE = "built-in"
    path = os.path.join(tok_dir or "", "generation_config.json")
    try:
        with open(path, encoding="utf-8") as f:
            gc = json.load(f)
    except (OSError, ValueError):
        return MODEL_FILTERS_SOURCE
    if not isinstance(gc, dict):
        return MODEL_FILTERS_SOURCE
    took = []
    k, p = gc.get("top_k"), gc.get("top_p")
    if isinstance(k, int) and not isinstance(k, bool) and k >= 0:
        MODEL_FILTERS["top_k"] = k
        took.append("top_k")
    if (isinstance(p, (int, float)) and not isinstance(p, bool)
            and 0.0 < p <= 1.0):
        MODEL_FILTERS["top_p"] = float(p)
        took.append("top_p")
    if took:
        MODEL_FILTERS_SOURCE = "generation_config.json"
    return MODEL_FILTERS_SOURCE

def filter_default(req, field):
    v = server_default(req, field)
    return v if v is not None else MODEL_FILTERS.get(field)

def defaults_summary():
    names = {"max_tokens": "HALOGEN_MAX_TOKENS_DEFAULT",
             "temperature": "HALOGEN_TEMPERATURE", "top_p": "HALOGEN_TOP_P",
             "top_k": "HALOGEN_TOP_K", "min_p": "HALOGEN_MIN_P",
             "presence_penalty": "HALOGEN_PRESENCE_PENALTY",
             "frequency_penalty": "HALOGEN_FREQUENCY_PENALTY",
             "repetition_penalty": "HALOGEN_REPETITION_PENALTY",
             "reasoning_effort": "HALOGEN_REASONING_EFFORT",
             "enable_thinking": "HALOGEN_ENABLE_THINKING",
             "max_thinking_tokens": "HALOGEN_MAX_THINKING_TOKENS",
             "thinking_answer_room": "HALOGEN_THINKING_ANSWER_ROOM"}
    return {names[k]: v for k, v in DEFAULTS.items() if v is not None}

class CompletionReq(BaseModel):
    model: str = MODEL_ID
    prompt: str = ""
    max_tokens: int = COMPLETION_MAX_TOKENS
    stream: bool = False
    stop: list[str] | str | None = None
    stream_options: dict | None = None
    drafter: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    logit_bias: dict | None = None
    logprobs: int | bool | None = None
    top_logprobs: int | None = None
    n: int | None = None
    response_format: dict | None = None

TEMPLATE_KWARGS = ("reasoning_effort", "enable_thinking", "preserve_thinking")
EOM = "<|im_end|>"

class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: list[dict]
    max_tokens: int = CHAT_MAX_TOKENS
    user: str | None = None
    prompt_cache_key: str | None = None
    safety_identifier: str | None = None
    metadata: dict | None = None
    max_completion_tokens: int | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    stream_options: dict | None = None
    drafter: str | None = None
    reasoning_effort: str | None = None
    max_thinking_tokens: int | None = None
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    thinking_budget_tokens: int | None = None
    thinking_budget: int | None = None
    thinking_token_budget: int | None = None
    reasoning: dict | None = None
    thinking: dict | str | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    chat_template_kwargs: dict | None = None
    continue_final_message: bool | None = None
    add_generation_prompt: bool | None = None

    @model_validator(mode="after")
    def _check_continuation(self):
        if self.add_generation_prompt is None and self.continue_final_message is None:
            return self
        cont = bool(self.continue_final_message)
        if self.add_generation_prompt is True and cont:
            raise ValueError(
                "continue_final_message and add_generation_prompt cannot both "
                "be true: one continues the final assistant message, the other "
                "opens a new one")
        if cont:
            last = self.messages[-1] if self.messages else None
            role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
            if role != "assistant":
                raise ValueError(
                    "continue_final_message needs the last message to be an "
                    "assistant message to continue; got "
                    + (f"role '{role}'" if role else "no messages"))
        return self

    @model_validator(mode="after")
    def _resolve_budget(self):
        sent = {}
        for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            if name in self.model_fields_set:
                v = getattr(self, name)
                if v is not None:
                    sent[name] = v
        distinct = set(sent.values())
        if len(distinct) > 1:
            raise ValueError(
                "conflicting token budgets ("
                + ", ".join(f"{k}={v}" for k, v in sorted(sent.items()))
                + "); they are the same budget under three names, send one")
        if distinct:
            self.max_tokens = distinct.pop()
        return self

    @model_validator(mode="after")
    def _fold_template_kwargs(self):
        if self.chat_template_kwargs is None:
            return self
        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")
        unknown = [k for k in self.chat_template_kwargs
                   if k not in TEMPLATE_KWARGS]
        if unknown:
            raise ValueError(
                "chat_template_kwargs contains unsupported key(s) "
                + ", ".join(repr(k) for k in sorted(unknown))
                + "; this server honours "
                + ", ".join(TEMPLATE_KWARGS)
                + ", which may also be sent as top-level fields")
        for k in TEMPLATE_KWARGS:
            if k not in self.chat_template_kwargs:
                continue
            nested = self.chat_template_kwargs[k]
            if k in self.model_fields_set:
                top = getattr(self, k)
                if top is not None and top != nested:
                    raise ValueError(
                        f"conflicting {k}: {top!r} at the top level and "
                        f"{nested!r} in chat_template_kwargs; they are the "
                        f"same control under two names, send one")
            setattr(self, k, nested)
        return self

    @model_validator(mode="after")
    def _fold_thinking_aliases(self):
        def take(field, name, value):
            if value is None:
                return
            cur = getattr(self, field)
            if cur is not None and cur != value:
                raise ValueError(
                    f"conflicting {field}: {cur!r} and {value!r} (as {name}); "
                    f"they are the same control under two names, send one")
            setattr(self, field, value)
        for name in ("thinking_budget_tokens", "thinking_budget", "thinking_token_budget"):
            v = getattr(self, name)
            if v is not None:
                if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                    raise ValueError(f"{name} must be a whole number >= 1")
                take("max_thinking_tokens", name, v)
        r = self.reasoning
        if r is not None:
            if not isinstance(r, dict):
                raise ValueError("reasoning must be an object "
                                 "({enabled, effort, max_tokens})")
            if r.get("effort") is not None:
                if not isinstance(r["effort"], str):
                    raise ValueError("reasoning.effort must be a string")
                take("reasoning_effort", "reasoning.effort", r["effort"])
            if r.get("max_tokens") is not None:
                v = r["max_tokens"]
                if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                    raise ValueError("reasoning.max_tokens must be a whole number >= 1")
                take("max_thinking_tokens", "reasoning.max_tokens", v)
            if r.get("enabled") is not None:
                if not isinstance(r["enabled"], bool):
                    raise ValueError("reasoning.enabled must be true or false")
                take("enable_thinking", "reasoning.enabled", r["enabled"])
        t = self.thinking
        if t is not None:
            if isinstance(t, str):
                take("reasoning_effort", "thinking", t)
            elif isinstance(t, dict):
                ty = t.get("type")
                if ty is not None:
                    if ty not in ("enabled", "disabled", "adaptive"):
                        raise ValueError("thinking.type must be enabled, disabled or adaptive")
                    if ty != "adaptive":
                        take("enable_thinking", "thinking.type", ty == "enabled")
                if t.get("budget_tokens") is not None:
                    v = t["budget_tokens"]
                    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                        raise ValueError("thinking.budget_tokens must be a whole number >= 1")
                    take("max_thinking_tokens", "thinking.budget_tokens", v)
            else:
                raise ValueError("thinking must be an object ({type, budget_tokens}) or a level")
        return self
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    logit_bias: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    n: int | None = None
    response_format: dict | None = None

class MessagesReq(BaseModel):
    model_config = {"extra": "allow"}
    model: str | None = None
    max_tokens: int
    messages: list
    system: str | list | None = None
    stream: bool = False
    tools: list | None = None
    tool_choice: dict | None = None
    thinking: dict | None = None
    output_config: dict | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict | None = None

class ResponsesReq(BaseModel):
    model: str = MODEL_ID
    instructions: str | None = None
    input: list[dict] | str = ""
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    drafter: str | None = None
    store: bool | None = None
    previous_response_id: str | None = None
    include: list[str] | None = None
    reasoning: dict | None = None
    prompt_cache_key: str | None = None
    safety_identifier: str | None = None
    metadata: dict | None = None
    client_metadata: dict | None = None
    service_tier: str | None = None
    user: str | None = None
    text: dict | None = None

    model_config = {"extra": "allow"}

def responses_tools(tools):
    if not tools:
        return None, []
    out, dropped = [], []

    def take(t):
        if t.get("type") != "function" or not t.get("name"):
            dropped.append(t.get("type") or "?")
            return
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object",
                                                  "properties": {}}}})

    for t in tools:
        if t.get("type") == "namespace":
            for sub in t.get("tools") or []:
                take(sub)
            continue
        take(t)
    return (out or None), dropped

def _item_parts(item):
    c = item.get("content")
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return ""
    parts, has_img = [], False
    for p in c:
        if isinstance(p, str):
            parts.append({"type": "text", "text": p})
            continue
        if not isinstance(p, dict):
            continue
        kind = p.get("type") or ""
        if not kind:
            kind = ("input_image" if ("image_url" in p or "image" in p)
                    else "input_text")
        if kind in ("input_image", "image_url", "image"):
            u = p.get("image_url", p.get("image"))
            if isinstance(u, dict):
                u = u.get("url")
            if not u:
                if p.get("file_id"):
                    raise ValueError(
                        "input_image.file_id is not supported: this server "
                        "has no file store. Inline the image as a data: URL.")
                raise ValueError("input_image carries no image_url")
            parts.append({"type": "image_url", "image_url": {"url": u}})
            has_img = True
        elif kind in ("input_file", "file"):
            raise ValueError("file inputs are not supported by this server")
        elif isinstance(p.get("text"), str):
            parts.append({"type": "text", "text": p["text"]})
    if not has_img:
        return "".join(p["text"] for p in parts if p["type"] == "text")
    return parts

_PART_TYPES = frozenset(("input_text", "output_text", "text", "input_image",
                         "image_url", "image", "input_file", "file"))

def _output_content(out):
    if isinstance(out, str):
        return out
    typed = [isinstance(p, dict) and isinstance(p.get("type"), str)
             and p["type"] in _PART_TYPES for p in out] \
        if isinstance(out, list) else []
    if any(typed) and all(t or isinstance(p, str) for t, p in zip(typed, out)):
        return _item_parts({"content": out})
    return json.dumps(out, ensure_ascii=False)

def responses_messages(instructions, items):
    sys_parts = []
    if instructions:
        sys_parts.append(instructions)
    msgs = []
    if isinstance(items, str):
        if sys_parts:
            msgs.append({"role": "system", "content": "\n\n".join(sys_parts)})
        if items:
            msgs.append({"role": "user", "content": items})
        return msgs

    for it in items or []:
        if not isinstance(it, dict):
            continue
        kind = it.get("type") or "message"
        if kind == "function_call":
            call = {"id": it.get("call_id") or it.get("id") or "",
                    "type": "function",
                    "function": {"name": it.get("name") or "",
                                 "arguments": it.get("arguments") or "{}"}}
            if (msgs and msgs[-1].get("role") == "assistant"
                    and isinstance(msgs[-1].get("content"), str)):
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "",
                             "tool_calls": [call]})
            continue
        if kind == "function_call_output":
            msgs.append({"role": "tool",
                         "tool_call_id": it.get("call_id") or "",
                         "content": _output_content(it.get("output"))})
            continue
        if kind in ("reasoning", "web_search_call", "item_reference"):
            continue
        role = it.get("role") or "user"
        content = _item_parts(it)
        if role in ("developer", "system"):
            if isinstance(content, list):
                raise ValueError("images are not supported in a system or "
                                 "developer turn; send the image in a user "
                                 "turn")
            if content:
                sys_parts.append(content)
            continue
        if (role == "assistant" and isinstance(content, str) and msgs
                and msgs[-1].get("role") == "assistant"
                and isinstance(msgs[-1].get("content"), str)):
            if content:
                prev = msgs[-1]["content"]
                msgs[-1]["content"] = prev + "\n\n" + content if prev else content
            continue
        if content or role == "assistant":
            msgs.append({"role": role, "content": content})
    if sys_parts:
        msgs.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})
    return msgs

SAMPLING_FIELDS = ("n",)

def penalty_spec(req):
    out = ""
    if req.logprobs:
        out += " LOGPROBS"
        if req.top_logprobs:
            out += " TOPLP %d" % int(req.top_logprobs)
    if sample_spec(req) is None:
        return out
    pp = float(server_default(req, "presence_penalty") or 0.0)
    fp = float(server_default(req, "frequency_penalty") or 0.0)
    if pp or fp:
        out += " PENALTY %.9g %.9g" % (pp, fp)
    rp = server_default(req, "repetition_penalty")
    if rp is not None and float(rp) != 1.0:
        out += " REPEAT %.9g" % float(rp)
    if req.logit_bias:
        items = []
        for k, v in req.logit_bias.items():
            try:
                tid = int(k)
            except (TypeError, ValueError):
                raise HTTPException(400, f"logit_bias key {k!r} is not a "
                                         f"token id")
            if not 0 <= tid < 248320:
                raise HTTPException(400, f"logit_bias token id {tid} is "
                                         f"outside 0..248319")
            items.append((tid, float(v)))
        if len(items) > 20480:
            raise HTTPException(400, f"logit_bias has {len(items)} entries, "
                                     f"over this server's cap of 20480")
        out += " BIAS %d " % len(items) + " ".join(
            "%d %.9g" % (i, v) for i, v in items)
    return out

def sample_spec(req):
    t = server_default(req, "temperature")
    if t is None or t <= 0.0:
        return None
    seed = req.seed if req.seed is not None else random.getrandbits(63)
    return (float(t), int(filter_default(req, "top_k") or 0),
            float(filter_default(req, "top_p") or 0.0),
            float(server_default(req, "min_p") or 0.0), int(seed))
_warned_sampling = set()
DRAFTERS = {"serial": 0, "mtp": 1, "dflash2": 2}
DRAFTER_CAP = {"mtp": "mtp", "dflash2": "dflash2"}

FORCE_OPEN = "<tool_call>\n<function="

def tool_choice_mode(req):
    tc = req.tool_choice
    if tc is None or tc == "auto":
        return "auto", None
    if isinstance(tc, str):
        if tc == "none":
            return "none", None
        if tc == "required":
            if not req.tools:
                raise HTTPException(400, "tool_choice 'required' needs tools")
            return "required", None
        raise HTTPException(400, f"tool_choice '{tc}' unknown; use "
                                 f"auto | none | required | "
                                 f"{{'type': 'function', ...}}")
    if isinstance(tc, dict) and tc.get("type") == "function":
        name = (tc.get("function") or {}).get("name")
        known = {(t.get("function") or t).get("name") for t in req.tools or []}
        if not isinstance(name, str):
            raise HTTPException(400, "tool_choice.function.name is required")
        if name not in known:
            raise HTTPException(400, f"tool_choice names '{name}', which is "
                                     f"not in tools")
        return "function", name
    raise HTTPException(400, "tool_choice must be a string or a "
                             "{'type': 'function', ...} object")

def force_prefill(mode, name, tools):
    if mode == "function":
        return FORCE_OPEN + name + ">\n"
    if mode == "required":
        names = [(t.get("function") or t).get("name") for t in tools or []]
        if len(names) == 1 and isinstance(names[0], str):
            return FORCE_OPEN + names[0] + ">\n"
        return FORCE_OPEN
    return ""

def wants_usage(req):
    so = req.stream_options
    return bool(isinstance(so, dict) and so.get("include_usage"))

def chat_kwargs(req, mode="auto"):
    kw = {}
    if req.reasoning_effort is not None and not effort_is_none(req):
        e = EFFORT_MAP.get(req.reasoning_effort.lower())
        if e is None:
            raise HTTPException(400, f"reasoning_effort "
                                     f"'{req.reasoning_effort}' unsupported; "
                                     f"use {'|'.join(EFFORT_VALUES)}")
        kw["reasoning_effort"] = e
    elif DEFAULTS["reasoning_effort"] is not None:
        kw["reasoning_effort"] = DEFAULTS["reasoning_effort"]
    if server_default(req, "enable_thinking") is not None:
        kw["enable_thinking"] = server_default(req, "enable_thinking")
    if effort_is_none(req):
        kw["enable_thinking"] = False
        kw.pop("reasoning_effort", None)
    if req.preserve_thinking is not None:
        kw["preserve_thinking"] = req.preserve_thinking
    if req.tools and mode != "none":
        kw["tools"] = req.tools
    return kw

class ThinkSplit:

    MARK = "</think>"

    def __init__(self, thinking=True):
        self.thinking = thinking
        self.full = ""
        self.sent_r = self.sent_c = 0
        self.cut = None

    def push(self, delta):
        self.full += delta
        if not self.thinking:
            out = self.full[self.sent_c:]
            self.sent_c = len(self.full)
            return "", out
        i = self.full.find(self.MARK)
        if i < 0 and self.cut is not None:
            r, c = self.full[:self.cut], self.full[self.cut:]
            rd, cd = r[self.sent_r:], c[self.sent_c:]
            self.sent_r, self.sent_c = len(r), len(c)
            return rd, cd
        if i < 0:
            hold = 0
            for k in range(len(self.MARK) - 1, 0, -1):
                if self.full.endswith(self.MARK[:k]):
                    hold = k
                    break
            r = self.full[:len(self.full) - hold]
            out = r[self.sent_r:]
            self.sent_r = len(r)
            return out, ""
        r, c = self.full[:i], self.full[i + len(self.MARK):]
        rd, cd = r[self.sent_r:], c[self.sent_c:]
        self.sent_r, self.sent_c = len(r), len(c)
        return rd, cd

    def parts(self):
        if not self.thinking:
            return "", self.full
        i = self.full.find(self.MARK)
        if i < 0 and self.cut is not None:
            return (self.full[:self.cut].strip(),
                    self.full[self.cut:].lstrip("\n"))
        if i < 0:
            return self.full.strip(), ""
        return (self.full[:i].strip(),
                self.full[i + len(self.MARK):].lstrip("\n"))

def common_prefix_len(a, b):
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo

class RecentPrefixes:

    def __init__(self, keep=16):
        self.items = collections.deque(maxlen=keep)

    def best(self, ids):
        a = array.array("i", ids)
        best, now = None, time.monotonic()
        for t, plen, seq in self.items:
            n = common_prefix_len(a, seq)
            if best is None or n > best[0]:
                best = (n, now - t, plen, len(seq))
        return best

    def add(self, ids, out):
        seq = array.array("i", ids)
        seq.extend(out)
        self.items.append((time.monotonic(), len(ids), seq))

def build_app(tok, engine, ctx, max_cap=4096, queue_timeout=600,
              template=None, npu=None):
    engine.tokenizer = tok
    load_model_filters(getattr(tok, "name_or_path", "") or "")
    engine.fallback_ctx = ctx
    app = FastAPI(title="halogen")

    RC = request_records.from_env(print)
    app.state.records = RC

    def rec_begin(wire, http_request, sess, src, n_msgs, model, stream,
                  effort=None, tool_choice=None):
        if RC is None:
            return None
        try:
            return RC.begin(wire, http_request, sess, src, n_msgs, model,
                            stream, effort, tool_choice)
        except Exception:
            return None

    INCREMENTAL_DETOK = not getattr(tok, "clean_up_tokenization_spaces", False)
    CARRY_MAX = 8

    def wire_error(request, status, message, code=None):
        if request.url.path.startswith("/v1/messages"):
            return anth_error(status, message)
        return oai_error(status, message, code)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc):
        return wire_error(request, exc.status_code, str(exc.detail),
                          "engine_busy" if exc.status_code == 503 else None)

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        return wire_error(request, 500, f"internal error: {type(exc).__name__}: "
                                        f"{str(exc)[:300]}", "server_error")

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):
        parts = []
        for e in exc.errors():
            loc = ".".join(str(x) for x in e.get("loc", ())
                           if x != "body") or "body"
            parts.append(f"{loc}: {e.get('msg', 'invalid')}")
        return wire_error(request, 400, "; ".join(parts) or "invalid request")
    eos_ids = [i for i in {tok.eos_token_id,
                           tok.convert_tokens_to_ids("<|im_end|>")}
               if isinstance(i, int) and i >= 0]

    def stop_list(stop):
        if stop is None:
            return []
        return [s for s in ([stop] if isinstance(stop, str) else list(stop)) if s]

    def stop_in(text, start, stops, kept_at):
        for s in stops:
            if not s:
                continue
            i = text.find(s, start)
            while i >= 0 and any(a < i + len(s) and i < b for a, b in kept_at):
                i = text.find(s, i + 1)
            if i >= 0:
                return s, i
        return None, -1

    def schema_for(req):
        rf = getattr(req, "response_format", None)
        if isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"):
            return schema_text(rf)
        return None

    def check_sampling(req):
        rf = getattr(req, "response_format", None)
        if isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"):
            if not engine.info.get("grammar"):
                why = engine.info.get("grammar_error")
                raise HTTPException(
                    400,
                    f"response_format (`text.format` on /v1/responses) "
                    f"{rf.get('type')!r} cannot be enforced here: "
                    + (f"the engine's vocabulary handshake failed ({why})"
                       if why else
                       f"the engine (version {engine.info.get('version', 'unknown')}) "
                       "has no constrained decoding; 0.8.0 and later do")
                    + ". /health reports `structured_output`. Omit the field "
                    "and parse the reply, or prompt for JSON and validate it.")
            if sample_spec(req) is not None:
                raise HTTPException(
                    400,
                    "response_format json_schema / json_object decodes greedy "
                    "in this release: send temperature 0 (or omit it). "
                    "Sampling under a schema is not implemented yet.")
            schema_text(rf)
        if sample_spec(req) is not None and not engine.info.get("sampling"):
            raise HTTPException(
                400,
                "this engine build decodes greedy only (a forward pass and an "
                "argmax); it does not sample. Omit `temperature` or send "
                "`temperature: 0`. /health reports what this build supports.")
        for f, lo, hi in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0),
                          ("min_p", 0.0, 1.0)):
            v = getattr(req, f)
            if v is not None and not (lo <= float(v) <= hi):
                raise HTTPException(
                    400, f"{f}={v} is outside {lo}..{hi}. Values past the "
                         f"range are NOT clamped, because a clamped request is "
                         f"indistinguishable, from the response, from one "
                         f"that was served as asked.")
        if req.top_k is not None and req.top_k < 0:
            raise HTTPException(400, f"top_k={req.top_k} must be >= 0 "
                                     f"(0 disables the filter).")
        if (req.logprobs or req.top_logprobs) and req.stream:
            raise HTTPException(
                400, "logprobs with stream=true is not implemented. The "
                     "logprob column is collected and returned with the "
                     "finished response. Send stream=false, or drop "
                     "logprobs.")
        tl = req.top_logprobs
        if tl is not None:
            if isinstance(tl, bool) or not isinstance(tl, int) or not 0 <= tl <= 20:
                raise HTTPException(
                    400, f"top_logprobs={tl!r} must be an integer in 0..20.")
            if tl and not req.logprobs:
                raise HTTPException(
                    400, "top_logprobs needs logprobs=true (the OpenAI rule): "
                         "send both.")
        scoring = bool(req.logprobs) and (sample_spec(req) is None or bool(tl))
        if scoring and req.max_tokens != 1:
            raise HTTPException(
                400, ("top_logprobs" if tl else "logprobs at temperature 0")
                     + " covers the first generated token only: send "
                     "max_tokens 1, which is the classifier shape (one "
                     "forward pass, the next-token distribution). With "
                     "temperature > 0 and no top_logprobs, every sampled "
                     "token carries its own logprob.")
        if req.logprobs and isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"):
            raise HTTPException(
                400, "logprobs with response_format is not supported: the "
                     "schema's mask picks the token, and an unmasked logprob "
                     "would not describe that pick. Drop one of the two.")
        rp = getattr(req, "repetition_penalty", None)
        if rp is not None and not (isinstance(rp, (int, float))
                                   and math.isfinite(rp) and rp > 0):
            raise HTTPException(400, f"repetition_penalty must be a finite "
                                     f"number greater than 0 (1 = off); got "
                                     f"{rp!r}")
        if sample_spec(req) is None:
            for f, why in (("presence_penalty", req.presence_penalty),
                           ("frequency_penalty", req.frequency_penalty),
                           ("repetition_penalty",
                            None if rp in (None, 1, 1.0) else rp),
                           ("logit_bias", req.logit_bias)):
                if why:
                    raise HTTPException(
                        400, f"{f} applies to the sampler, and this request "
                             f"decodes greedy (temperature absent or 0). "
                             f"Send temperature > 0, or drop {f}: a request "
                             f"silently served without it would be "
                             f"indistinguishable from one that honoured it.")

    def drafter_for(req):
        return drafter_id(req.drafter)

    def drafter_id(name):
        if name is None:
            return None
        key = name.lower()
        v = DRAFTERS.get(key)
        if v is None:
            raise HTTPException(400, f"drafter '{name}' unknown; use "
                                     f"{' | '.join(DRAFTERS)}")
        cap = DRAFTER_CAP.get(key)
        if cap is not None and not engine.info.get(cap):
            raise HTTPException(400, f"drafter '{key}' unavailable: the "
                                     f"loaded checkpoint cannot serve it")
        return v

    THINK_END_ID = tok.convert_tokens_to_ids(ThinkSplit.MARK) \
        if hasattr(tok, "convert_tokens_to_ids") else None
    if not isinstance(THINK_END_ID, int) or THINK_END_ID < 0:
        THINK_END_ID = None
    msg = os.environ.get("HALOGEN_THINKING_BUDGET_MESSAGE")
    if msg is None:
        msg = QWEN_BUDGET_SENTENCE
    msg_ids = (tok("\n\n" + msg + "\n", add_special_tokens=False)["input_ids"]
               if msg else tok("\n\n", add_special_tokens=False)["input_ids"])
    control = set(getattr(tok, "all_special_ids", None) or [])
    for name in ("<think>", "</think>", "<tool_call>", "</tool_call>"):
        i = tok.convert_tokens_to_ids(name) \
            if hasattr(tok, "convert_tokens_to_ids") else None
        if isinstance(i, int) and i >= 0:
            control.add(i)
    hit = [tok.decode([i]) for i in msg_ids if i in control]
    if hit:
        raise SystemExit(f"serve_api: HALOGEN_THINKING_BUDGET_MESSAGE contains "
                         f"{', '.join(sorted(set(hit)))}, which the model would "
                         f"read as a control token; write the message as "
                         f"plain text")
    THINK_CLOSE_IDS = (
        msg_ids
        + ([THINK_END_ID] if THINK_END_ID is not None else [])
        + tok("\n\n", add_special_tokens=False)["input_ids"])
    if len(THINK_CLOSE_IDS) > THINK_CLOSE_MAX:
        raise SystemExit(f"serve_api: HALOGEN_THINKING_BUDGET_MESSAGE is "
                         f"{len(THINK_CLOSE_IDS)} tokens with the closing "
                         f"marker; the engine takes at most {THINK_CLOSE_MAX}")
    TOOL_CALL_ID = tok.convert_tokens_to_ids("<tool_call>") \
        if hasattr(tok, "convert_tokens_to_ids") else None
    if not isinstance(TOOL_CALL_ID, int) or TOOL_CALL_ID < 0:
        TOOL_CALL_ID = None

    def _one_id(s):
        i = tok.convert_tokens_to_ids(s) \
            if hasattr(tok, "convert_tokens_to_ids") else None
        return i if isinstance(i, int) and i >= 0 else None
    IM_START_ID = _one_id("<|im_start|>")
    THINK_OPEN_ID = _one_id("<think>")
    TOOL_CALL_END_ID = _one_id("</tool_call>")
    EOS_SET = frozenset(eos_ids)
    LITERAL = {i: s for i, s in ((IM_START_ID, "<|im_start|>"),
                                 (_one_id(EOM), EOM)) if i is not None}

    def prompt_state(ids):
        if IM_START_ID is None:
            return None
        k = len(ids) - 1
        while k >= 0 and ids[k] != IM_START_ID:
            k -= 1
        if k < 0:
            return None
        in_think = in_call = False
        for t in ids[k + 1:]:
            if in_think:
                if t == THINK_END_ID:
                    in_think = False
            elif t == THINK_OPEN_ID:
                in_think = True
            elif t == TOOL_CALL_ID:
                in_call = True
            elif t == TOOL_CALL_END_ID:
                in_call = False
        if THINK_END_ID is None:
            in_think = False
        return in_think, in_call

    def guard_for(ids, tools=True):
        if not EOS_GUARD or IM_START_ID is None or not EOS_SET:
            return None
        st = prompt_state(ids)
        if st is None:
            return None
        in_think, in_call = st
        if not tools:
            return (int(in_think), 0,
                    -1 if THINK_END_ID is None else THINK_END_ID, -1, -1)
        return (int(in_think), int(in_call),
                -1 if THINK_END_ID is None else THINK_END_ID,
                -1 if TOOL_CALL_ID is None else TOOL_CALL_ID,
                -1 if TOOL_CALL_END_ID is None else TOOL_CALL_END_ID)

    recent = RecentPrefixes()

    async def run(ids, max_tokens, stops, drafter=None, sample=None,
                  penalty="", snap=0, snap2=0, images=None, schema=None,
                  after=None, escape=(), think=None, seg=None, snap3=0,
                  think_on=None, think_src=None, guard=None):
        limit = engine.info.get("ctx") or ctx
        if len(ids) >= limit:
            raise HTTPException(400, f"prompt {len(ids)} tokens exceeds "
                                     f"context {limit}")
        out, text, done = [], "", None
        lps = []

        def think_end(dd):
            if THINK_END_ID is not None and THINK_END_ID in out:
                dd["think_end"] = out.index(THINK_END_ID) + 1
            else:
                dd["think_end"] = 0
            return dd
        tail_ids, tail_text = [], ""
        use_window = INCREMENTAL_DETOK
        prof = {"detok": 0.0, "wait": 0.0, "n": 0}
        lit = LITERAL if guard else {}

        def detok(seq):
            if not lit or not any(t in lit for t in seq):
                return tok.decode(seq, skip_special_tokens=True)
            parts, run_ = [], []
            for t in seq:
                if t in lit:
                    if run_:
                        parts.append(tok.decode(run_, skip_special_tokens=True))
                        run_ = []
                    parts.append(lit[t])
                else:
                    run_.append(t)
            if run_:
                parts.append(tok.decode(run_, skip_special_tokens=True))
            return "".join(parts)

        async def held(stream):
            pending = None
            async for tid, d, lp in stream:
                if pending is not None:
                    yield pending[0], None, pending[1], d is None
                    pending = None
                if d is None and tid in EOS_SET:
                    pending = (tid, lp)
                    continue
                yield tid, d, lp, False
            if pending is not None:
                yield pending[0], None, pending[1], False

        async def plain(stream):
            async for tid, d, lp in stream:
                yield tid, d, lp, False

        _t_last = time.perf_counter()
        pmatch = (recent.best(ids) if not images
                  and engine.info.get("cache_mode", 1) != 0 else None)
        ctl = {"stop": False}
        stop_n = None
        stop_seq = None
        sent = 0
        kept_at = []
        stream = engine.generate(ids, max_tokens, eos_ids, drafter, sample,
                                 penalty, snap=snap, snap2=snap2,
                                 images=images, schema=schema, after=after,
                                 escape=escape, think=think, seg=seg,
                                 snap3=snap3, guard=guard, ctl=ctl)
        async for tid, d, lp, kept in (held(stream) if guard else plain(stream)):
            if stop_n is not None and d is None:
                continue
            if lp is not None:
                lps.append((tid, lp))
            if d is not None:
                if stop_n is not None:
                    n_eng = int(d.get("n_gen", 0))
                    if n_eng > stop_n > 0:
                        d["decode_ms"] = d["decode_ms"] * stop_n / n_eng
                    d["n_gen"] = stop_n
                    d["reason"] = "stop"
                done = d
                prefix_s = ""
                if d.get("reason") != "error":
                    if pmatch is not None:
                        shared, age, plen, slen = pmatch
                        d["prefix_n"] = shared
                        if int(d.get("n_prompt", 0)) - int(d.get("n_cached", 0)) \
                                >= PREFIX_NOTE_MIN:
                            where = ("all of it" if shared >= slen else
                                     "then its prompt differs" if shared < plen else
                                     "then its reply differs")
                            prefix_s = (f" | closest earlier: {shared} tokens shared "
                                        f"with a request {age:.0f}s ago, {where}")
                    recent.add(ids, out)
                if d.get("think_forced") == 2:
                    d["think_closed_by"] = "end_of_turn_guard"
                elif think_src and d.get("think_forced"):
                    d["think_closed_by"] = think_src
                if d.get("reason") != "error":
                    engine.metrics.record(timings(d), structured=bool(schema), d=d)
                dt = d["decode_ms"] / 1000
                name = {1: "mtp", 2: "dflash2"}.get(d.get("drafter"),
                                                    "spec" if d.get("rounds")
                                                    else "batch")
                spec = ""
                shared = d.get("shared", 0)
                if d.get("rounds"):
                    spec = (f"{d['rounds']} rounds, commit "
                            f"{(d['commit'] - shared) / d['rounds']:.2f}"
                            f"/round | ")
                pld = ""
                if d.get("pld_rounds"):
                    pld = (f" | pld {d['pld_rounds']} rounds, "
                           f"{d['pld_accepted'] / d['pld_rounds']:.2f}"
                           f" acc/round")
                rate = (f"{d['n_gen'] / dt:.2f} t/s"
                        if dt > 0 and d["n_gen"] > 1 else "n/a")
                c = d.get("n_cached")
                pn = d["n_prompt"]
                cached_s = (f" ({c} cached, {100.0 * c / pn:.1f}%)" if c and pn else
                            f" ({c} cached)" if c else "")
                pf_s = d["prefill_ms"] / 1000
                pf_n = max(0, pn - (c or 0))
                pf_rate = (f" = {pf_n / pf_s:.0f} t/s" if pf_s > 0 and pf_n >= PREFILL_RATE_MIN_TOKENS
                           else f" ({pf_n} new)" if pf_n > 0 else "")
                pool_s = (f" | pool {d['pool_used']}/{d['pool_positions']}"
                          f" {100.0 * d['pool_used'] / d['pool_positions']:.0f}%"
                          if d.get("pool_positions") else "")
                clamp_s = (f" | max_tokens clamped {d['room_from']} -> {d['max_tokens']}"
                           if d.get("room_from", -1) >= 0 else "")
                think_s = ("" if think_on is None else
                           f" | think {'on' if think_on else 'off'}"
                           + (f", closed at {d.get('think_seen', 0)} by "
                              + {"answer_room": "answer room",
                                 "end_of_turn_guard": "the end-of-turn guard"}.get(
                                     d["think_closed_by"], d["think_closed_by"])
                              if d.get("think_closed_by") else ""))
                kt, kc = d.get("kept_think", 0), d.get("kept_call", 0)
                think_s += ((f" | end of turn kept as text: {kt} in the think "
                             f"block, {kc} in a tool call") if kt or kc else "")
                if d.get("disk_rows"):
                    think_s += (f" | disk: restored {d['disk_rows']} rows in "
                                f"{d.get('disk_ms', 0.0):.0f} ms")
                print(f"serve_api: {name} {d['n_gen']} tok in {dt:.2f}s = "
                      f"{rate} | {spec}"
                      f"prompt {pn}{cached_s}"
                      f", prefill {pf_s:.2f}s{pf_rate}"
                      f" | detok {prof['detok'] / max(prof['n'], 1) * 1e6:.0f}us/tok"
                      + (f" | {shared} tok with the head off" if shared
                         else "")
                      + pld + pool_s + clamp_s + think_s + prefix_s,
                      flush=True)
                break
            _t0 = time.perf_counter()
            prof["wait"] += _t0 - _t_last
            out.append(tid)
            if lit and tid in EOS_SET and not kept:
                yield "", None
                continue
            windowed = use_window
            if windowed:
                win_ids = tail_ids + [tid]
                new, base = detok(win_ids), tail_text
            else:
                new, base = detok(out), text
            prof["detok"] += time.perf_counter() - _t0
            prof["n"] += 1
            holding = new.endswith("�")
            while new.endswith("�"):
                new = new[:-1]
            if windowed and not new.startswith(base):
                new = detok(out)
                while new.endswith("�"):
                    new = new[:-1]
                base, tail_ids, tail_text = text, [], ""
                holding, windowed, use_window = False, False, False
            if not new.startswith(base):
                n = 0
                while n < len(base) and n < len(new) and base[n] == new[n]:
                    n += 1
                delta = new[n:]
                text = text + delta if windowed else new
            else:
                delta = new[len(base):]
                text = text + delta if windowed else new
            if windowed:
                if holding and len(win_ids) < CARRY_MAX:
                    tail_ids, tail_text = win_ids, new
                else:
                    tail_ids, tail_text = [], ""
            if kept and tid in lit and text.endswith(lit[tid]):
                kept_at.append((len(text) - len(lit[tid]), len(text)))
            if stops and delta:
                span = max(len(t) for t in stops if t) - 1 + len(delta)
                win = text[-span:] if 0 < span < len(text) else text
                hit, cut = stop_in(text, len(text) - len(win), stops, kept_at)
            else:
                win, hit = "", None
            if hit:
                tail = text[min(sent, cut):cut]
                if tail:
                    yield tail, None
                stop_n = len(out)
                stop_seq = hit
                ctl["stop"] = True
                continue
            if stops:
                hold = 0
                for t in stops:
                    for k in range(min(len(t) - 1, len(text)), hold, -1):
                        if text.endswith(t[:k]):
                            hold = k
                            break
                end = len(text) - hold
                delta = text[sent:end] if end > sent else ""
                sent = max(sent, end)
            if delta:
                _t_last = time.perf_counter()
            yield delta, None
        if stops and stop_n is None and sent < len(text):
            yield text[sent:], None
        if done is None and stop_n is not None:
            done = {"reason": "stop", "n_gen": stop_n, "n_prompt": len(ids),
                    "prefill_ms": 0.0, "decode_ms": 0.0}
        if done is not None:
            done["logprobs"] = lps
            if stop_seq is not None:
                done["stop_seq"] = stop_seq
        yield None, think_end(done or {"reason": "length", "n_gen": len(out),
                                       "n_prompt": len(ids), "prefill_ms": 0.0,
                                       "decode_ms": 0.0, "logprobs": lps})

    def reasoning_tokens(d, thinking):
        if thinking is False:
            return 0
        if d.get("think_end"):
            return min(int(d["think_end"]), int(d.get("n_gen", 0)))
        return int(d.get("n_gen", 0)) if thinking else 0

    def usage(d, n_prompt):
        u = {"prompt_tokens": n_prompt, "completion_tokens": d["n_gen"],
             "total_tokens": n_prompt + d["n_gen"]}
        if d.get("n_cached"):
            u["prompt_tokens_details"] = {"cached_tokens": d["n_cached"]}
        if "n_reasoning" in d:
            det = {"reasoning_tokens": d["n_reasoning"]}
            if d.get("think_closed_by"):
                det["reasoning_closed_at"] = int(d.get("think_seen", 0))
                det["reasoning_closed_by"] = d["think_closed_by"]
            u["completion_tokens_details"] = det
        kept = int(d.get("kept_think", 0)) + int(d.get("kept_call", 0))
        if kept:
            u.setdefault("completion_tokens_details", {})["end_of_turn_kept"] = kept
        if d.get("call_in_block"):
            u.setdefault("completion_tokens_details", {})["tool_call_from_reasoning"] = True
        return u

    def responses_usage(d, n_prompt):
        u = usage(d, n_prompt)
        r = {"input_tokens": u.get("prompt_tokens", 0),
             "output_tokens": u.get("completion_tokens", 0),
             "total_tokens": u.get("total_tokens", 0)}
        if "prompt_tokens_details" in u:
            r["input_tokens_details"] = dict(u["prompt_tokens_details"])
        if "completion_tokens_details" in u:
            r["output_tokens_details"] = dict(u["completion_tokens_details"])
        return r

    def timings(d):
        pn = int(d.get("n_prompt", 0))
        gn = int(d.get("n_gen", 0))
        cached = int(d.get("n_cached", 0))
        processed = max(0, pn - cached)
        pms = float(d.get("prefill_ms", 0.0))
        dms = float(d.get("decode_ms", 0.0))
        t = {"prompt_n": processed, "predicted_n": gn,
             "prompt_ms": round(pms, 3), "predicted_ms": round(dms, 3),
             "prompt_per_second": (round(processed / (pms / 1000.0), 3)
                                   if pms > 0 and processed > 0 else 0.0),
             "predicted_per_second": round(gn / (dms / 1000.0), 3) if dms > 0 else 0.0}
        if "n_cached" in d:
            t["cache_n"] = cached
        if "disk_rows" in d:
            t["disk_restore_n"] = int(d["disk_rows"])
            t["disk_restore_ms"] = round(float(d["disk_ms"]), 3)
        if "prefix_n" in d:
            t["prefix_n"] = int(d["prefix_n"])
        if int(d.get("room_from", -1)) >= 0:
            t["max_tokens_clamped_from"] = int(d["room_from"])
            t["max_tokens_clamped_to"] = int(d["max_tokens"])
        if "rounds" in d:
            pld_r = int(d.get("pld_rounds", 0))
            pld_a = int(d.get("pld_accepted", 0))
            head_r = int(d["rounds"]) - pld_r
            head_a = int(d["commit"]) - int(d["rounds"]) - int(d.get("shared", 0)) - pld_a
            if pld_r == 0 or "pld_drafted" in d:
                head_d = int(d["head_drafted"]) if "head_drafted" in d else max(0, head_r)
                t["draft_n"] = head_d + int(d.get("pld_drafted", 0))
                t["draft_n_accepted"] = max(0, head_a) + pld_a
        return t

    async def keepalive(frames, every):
        if every <= 0:
            async for f in frames:
                yield f
            return
        it = frames.__aiter__()
        nxt = None
        try:
            while True:
                if nxt is None:
                    nxt = asyncio.ensure_future(it.__anext__())
                done, _ = await asyncio.wait({nxt}, timeout=every)
                if not done:
                    yield ": keepalive\n\n"
                    continue
                task, nxt = nxt, None
                try:
                    f = task.result()
                except StopAsyncIteration:
                    return
                yield f
        finally:
            if nxt is not None and not nxt.done():
                nxt.cancel()
                try:
                    await nxt
                except BaseException:
                    pass

    async def until_gone(frames, http_request, state, every=0.5):
        if http_request is None:
            async for f in frames:
                yield f
            return
        it = frames.__aiter__()
        nxt = None
        try:
            while True:
                if nxt is None:
                    nxt = asyncio.ensure_future(it.__anext__())
                done, _ = await asyncio.wait({nxt}, timeout=every)
                if not done:
                    if await http_request.is_disconnected():
                        state["gone"] = True
                        return
                    continue
                task, nxt = nxt, None
                try:
                    f = task.result()
                except StopAsyncIteration:
                    return
                yield f
        finally:
            if nxt is not None and not nxt.done():
                nxt.cancel()
                try:
                    await nxt
                except BaseException:
                    pass

    async def guarded(gen):
        try:
            async for item in gen:
                yield item
        except HTTPException as e:
            print(f"serve_api: error after the stream started: "
                  f"{e.status_code} {e.detail}", flush=True)
            yield None, {"reason": "error", "status": e.status_code,
                         "message": str(e.detail),
                         "code": ("engine_busy" if e.status_code == 503
                                  else None)}

    async def call_in_block(gen, split, tools, head=""):
        acc = ""
        try:
            async for delta, d in gen:
                if d is None:
                    acc += delta
                    yield delta, d
                    continue
                body = (head + acc).rstrip()
                i = body.rfind(TC_OPEN)
                if (d.get("reason") == "stop" and ThinkSplit.MARK not in acc
                        and body.endswith(TC_CLOSE) and i >= 0
                        and (i >= len(head) or TC_CLOSE not in head[i:])
                        and split_tool_calls(body[i:], tools)[1]):
                    split.cut = len(acc)
                    d["call_in_block"] = 1
                    print("serve_api: the reply ended inside its think block "
                          "right after a complete tool call; that call is "
                          "made (the end-of-turn guard)", flush=True)
                    yield body[i:], None
                yield delta, d
        finally:
            await gen.aclose()

    def error_body(d):
        return {"message": d["message"],
                "type": ERROR_TYPES.get(d["status"], "server_error"),
                "param": None, "code": d.get("code")}

    async def sse(gen, cid, created, chat, split, tstream=None, pre="",
                  parallel=True, include_usage=False):
        def envelope():
            return {"id": cid, "created": created, "model": MODEL_ID,
                    "object": "chat.completion.chunk" if chat
                              else "text_completion"}

        def frame(payload, fin=None):
            f = envelope()
            f["choices"] = [{"index": 0, "finish_reason": fin, **payload}]
            if include_usage:
                f["usage"] = None
            return f

        buf = ""
        c_started = False
        rd_started = False
        rd_pending = ""
        role_due = chat
        async for delta, d in gen:
            _s0 = time.perf_counter()
            if d is not None and d.get("reason") == "error":
                yield f"data: {json.dumps({'error': error_body(d)})}\n\n"
                yield "data: [DONE]\n\n"
                return
            if role_due:
                role_due = False
                yield (f"data: {json.dumps(frame({'delta': {'role': 'assistant', 'content': ''}}))}"
                       f"\n\n")
            if d is not None:
                fin = {"stop": "stop", "cancel": "stop",
                       "length": "length"}.get(d["reason"], "stop")
                if tstream is not None and tstream.any_calls() \
                        and fin != "length":
                    fin = "tool_calls"
                d["n_reasoning"] = reasoning_tokens(d, split.thinking if chat else None)
                last = frame({'delta': {}} if chat else {'text': ''}, fin)
                last["timings"] = timings(d)
                yield f"data: {json.dumps(last)}\n\n"
                if include_usage:
                    u = envelope()
                    u["choices"] = []
                    u["usage"] = usage(d, d.get("n_prompt", 0))
                    u["timings"] = timings(d)
                    yield f"data: {json.dumps(u)}\n\n"
                yield "data: [DONE]\n\n"
                return
            if not chat:
                _ev = f"data: {json.dumps(frame({'text': delta}))}\n\n"
                SSE_PROF["t"] += time.perf_counter() - _s0
                SSE_PROF["n"] += 1
                yield _ev
                continue
            rd, cd = split.push(delta)
            SSE_PROF["n"] += 1
            SSE_PROF["t"] += time.perf_counter() - _s0
            if rd:
                if not rd_started:
                    rd = rd.lstrip()
                    if rd:
                        rd_started = True
                if rd_started and rd:
                    rd = rd_pending + rd
                    kept = rd.rstrip()
                    rd_pending = rd[len(kept):]
                    rd = kept
                if rd:
                    yield (f"data: "
                           f"{json.dumps(frame({'delta': {'reasoning_content': rd}}))}"
                           f"\n\n")
            if not cd:
                continue
            if not c_started:
                cd = cd.lstrip("\n")
                if not cd:
                    continue
                c_started = True
            if tstream is None:
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'content': cd}}))}\n\n")
                continue
            buf += cd
            td, evs = tstream.push(pre + buf)
            if td:
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'content': td}}))}\n\n")
            for e in evs:
                if not parallel and e["index"] > 0:
                    continue
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'tool_calls': [e]}}))}"
                       f"\n\n")

    def messages_usage(d, n_prompt):
        cached = int(d.get("n_cached") or 0)
        return {"input_tokens": max(0, int(n_prompt) - cached),
                "output_tokens": int(d.get("n_gen") or 0),
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0}

    def messages_stop(d, called):
        if d.get("reason") == "length":
            return "max_tokens", None
        if called:
            return "tool_use", None
        if d.get("stop_seq"):
            return "stop_sequence", d["stop_seq"]
        return "end_turn", None

    def tool_input(args):
        try:
            v = json.loads(args or "{}")
        except ValueError:
            return {}
        return v if isinstance(v, dict) else {}

    async def sse_messages(gen, mid, model, split, tstream=None, pre="",
                           parallel=True, n_prompt=0, show_thinking=True):
        def ev(kind, **body):
            return "event: %s\ndata: %s\n\n" % (kind, json.dumps(
                dict({"type": kind}, **body)))

        yield ev("message_start", message={
            "id": mid, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": int(n_prompt), "output_tokens": 0,
                      "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}})
        nxt = itertools.count(0)
        cur = None
        rs_text, rs_started, rs_pending = "", False, ""
        started, buf = False, ""
        calls = {}

        def close_cur():
            nonlocal cur
            if cur is None:
                return ""
            kind, i = cur
            cur = None
            out = ""
            if kind == "thinking":
                out = ev("content_block_delta", index=i, delta={
                    "type": "signature_delta",
                    "signature": think_signature(rs_text)})
            return out + ev("content_block_stop", index=i)

        def open_block(kind, block):
            nonlocal cur
            cur = (kind, next(nxt))
            return ev("content_block_start", index=cur[1], content_block=block)

        async for delta, d in gen:
            if d is not None and d.get("reason") == "error":
                yield ev("error", error=anth_error_obj(d["status"],
                                                       d["message"]))
                return
            if d is not None:
                out = close_cur()
                if out:
                    yield out
                d["n_reasoning"] = reasoning_tokens(d, split.thinking)
                sr, ss = messages_stop(d, bool(calls))
                u = messages_usage(d, d.get("n_prompt") or n_prompt)
                yield ev("message_delta",
                         delta={"stop_reason": sr, "stop_sequence": ss},
                         usage=u)
                yield ev("message_stop")
                return
            rd, cd = split.push(delta)
            if rd:
                if not rs_started:
                    rd = rd.lstrip()
                    if rd:
                        rs_started = True
                if rs_started and rd:
                    rd = rs_pending + rd
                    kept = rd.rstrip()
                    rs_pending = rd[len(kept):]
                    rd = kept
                if rd:
                    if cur is None or cur[0] != "thinking":
                        yield close_cur()
                        yield open_block("thinking", {"type": "thinking",
                                                      "thinking": ""})
                    rs_text += rd
                    if show_thinking:
                        yield ev("content_block_delta", index=cur[1], delta={
                            "type": "thinking_delta", "thinking": rd})
            if not cd:
                continue
            if cur is not None and cur[0] == "thinking":
                yield close_cur()
            if not started:
                cd = cd.lstrip("\n")
                if not cd:
                    continue
                started = True
            td, evs = cd, []
            if tstream is not None:
                buf += cd
                td, evs = tstream.push(pre + buf)
            if td:
                if cur is None or cur[0] != "text":
                    yield close_cur()
                    yield open_block("text", {"type": "text", "text": ""})
                yield ev("content_block_delta", index=cur[1], delta={
                    "type": "text_delta", "text": td})
            for e in evs:
                i = e["index"]
                if not parallel and i > 0:
                    continue
                fn = e.get("function") or {}
                if i not in calls:
                    yield close_cur()
                    yield open_block("tool", {"type": "tool_use",
                                              "id": e.get("id") or "",
                                              "name": fn.get("name") or "",
                                              "input": {}})
                    calls[i] = cur[1]
                frag = fn.get("arguments") or ""
                if frag:
                    yield ev("content_block_delta", index=calls[i],
                             delta={"type": "input_json_delta",
                                    "partial_json": frag})

    async def sse_responses(gen, rid, created, model, split,
                            tstream=None, pre="", parallel=True,
                            max_out=0, want_summary=False):
        seq = itertools.count(0)

        def ev(kind, **body):
            body["type"] = kind
            body["sequence_number"] = next(seq)
            return f"event: {kind}\ndata: {json.dumps(body)}\n\n"

        def envelope(status, output, usage_obj=None):
            return {"id": rid, "object": "response", "created_at": created,
                    "status": status, "model": model, "output": output,
                    "parallel_tool_calls": parallel,
                    "max_output_tokens": max_out or None,
                    "usage": usage_obj}

        base = envelope("in_progress", [])
        yield ev("response.created", response=base)
        yield ev("response.in_progress", response=base)

        nxt = itertools.count(0)
        started = False
        msg_id = "msg_" + rid.split("_", 1)[-1]
        msg_open = False
        msg_index = None
        text_out = ""
        buf = ""
        calls = {}
        done_items = []

        def open_message():
            nonlocal msg_open, msg_index
            msg_index = next(nxt)
            msg_open = True
            item = {"id": msg_id, "type": "message", "role": "assistant",
                    "status": "in_progress", "content": []}
            return (ev("response.output_item.added", output_index=msg_index,
                       item=item)
                    + ev("response.content_part.added", item_id=msg_id,
                         output_index=msg_index, content_index=0,
                         part={"type": "output_text", "text": "",
                               "annotations": []}))

        def close_message():
            item = {"id": msg_id, "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text_out,
                                 "annotations": []}]}
            done_items.append((msg_index, item))
            return (ev("response.output_text.done", item_id=msg_id,
                       output_index=msg_index, content_index=0, text=text_out)
                    + ev("response.content_part.done", item_id=msg_id,
                         output_index=msg_index, content_index=0,
                         part={"type": "output_text", "text": text_out,
                               "annotations": []})
                    + ev("response.output_item.done", output_index=msg_index,
                         item=item))

        rs_id = "rs_" + rid.split("_", 1)[-1]
        rs_open = False
        rs_index = None
        rs_text = ""
        rs_pending = ""
        rs_started = False

        def open_reasoning():
            nonlocal rs_open, rs_index
            rs_index = next(nxt)
            rs_open = True
            item = {"id": rs_id, "type": "reasoning", "status": "in_progress",
                    "summary": [], "content": [], "encrypted_content": None}
            out = ev("response.output_item.added", output_index=rs_index,
                     item=item)
            if want_summary:
                out += ev("response.reasoning_summary_part.added",
                          item_id=rs_id, output_index=rs_index,
                          summary_index=0,
                          part={"type": "summary_text", "text": ""})
            return out

        def reasoning_delta(rd):
            nonlocal rs_text
            rs_text += rd
            if want_summary:
                return ev("response.reasoning_summary_text.delta",
                          item_id=rs_id, output_index=rs_index,
                          summary_index=0, delta=rd)
            return ev("response.reasoning_text.delta", item_id=rs_id,
                      output_index=rs_index, content_index=0, delta=rd)

        def close_reasoning():
            nonlocal rs_open
            rs_open = False
            item = {"id": rs_id, "type": "reasoning", "status": "completed",
                    "summary": ([{"type": "summary_text", "text": rs_text}]
                                if want_summary else []),
                    "content": [{"type": "reasoning_text", "text": rs_text}],
                    "encrypted_content": None}
            done_items.append((rs_index, item))
            if want_summary:
                out = ev("response.reasoning_summary_text.done",
                         item_id=rs_id, output_index=rs_index,
                         summary_index=0, text=rs_text)
                out += ev("response.reasoning_summary_part.done",
                          item_id=rs_id, output_index=rs_index,
                          summary_index=0,
                          part={"type": "summary_text", "text": rs_text})
            else:
                out = ev("response.reasoning_text.done", item_id=rs_id,
                         output_index=rs_index, content_index=0, text=rs_text)
            out += ev("response.output_item.done", output_index=rs_index,
                      item=item)
            return out

        async for delta, d in gen:
            if d is not None and d.get("reason") == "error":
                yield ev("error", **error_body(d))
                return
            if d is not None:
                if rs_open:
                    yield close_reasoning()
                if msg_open:
                    yield close_message()
                for st in calls.values():
                    if st["closed"]:
                        continue
                    st["closed"] = True
                    item = {"id": st["fc_id"], "type": "function_call",
                            "status": "completed", "name": st["name"],
                            "arguments": st["args"], "call_id": st["call_id"]}
                    done_items.append((st["index"], item))
                    yield ev("response.function_call_arguments.done",
                             item_id=st["fc_id"], output_index=st["index"],
                             arguments=st["args"])
                    yield ev("response.output_item.done",
                             output_index=st["index"], item=item)
                status = "incomplete" if d.get("reason") == "length" \
                    else "completed"
                out = [i for _, i in sorted(done_items, key=lambda x: x[0])]
                d["n_reasoning"] = reasoning_tokens(d, split.thinking)
                final = envelope(status, out, responses_usage(d, d.get("n_prompt", 0)))
                if status == "incomplete":
                    final["incomplete_details"] = {"reason": "max_output_tokens"}
                    yield ev("response.incomplete", response=final, timings=timings(d))
                else:
                    yield ev("response.completed", response=final, timings=timings(d))
                return

            rd, cd = split.push(delta)
            if rd:
                if not rs_started:
                    rd = rd.lstrip()
                    if rd:
                        rs_started = True
                if rs_started and rd:
                    rd = rs_pending + rd
                    kept = rd.rstrip()
                    rs_pending = rd[len(kept):]
                    rd = kept
                if rd:
                    if not rs_open:
                        yield open_reasoning()
                    yield reasoning_delta(rd)
            if not cd:
                continue
            if rs_open:
                yield close_reasoning()
            if not started:
                cd = cd.lstrip("\n")
                if not cd:
                    continue
                started = True
            if tstream is None:
                if not msg_open:
                    yield open_message()
                text_out += cd
                yield ev("response.output_text.delta", item_id=msg_id,
                         output_index=msg_index, content_index=0, delta=cd)
                continue
            buf += cd
            td, evs = tstream.push(pre + buf)
            if td:
                if not msg_open:
                    yield open_message()
                text_out += td
                yield ev("response.output_text.delta", item_id=msg_id,
                         output_index=msg_index, content_index=0, delta=td)
            for e in evs:
                i = e["index"]
                if not parallel and i > 0:
                    continue
                fn = e.get("function") or {}
                if i not in calls:
                    if msg_open:
                        yield close_message()
                        msg_open = False
                    idx = next(nxt)
                    calls[i] = {"index": idx, "fc_id": f"fc_{uuid.uuid4().hex[:24]}",
                                "call_id": e.get("id") or "",
                                "name": fn.get("name") or "", "args": "",
                                "closed": False}
                    st = calls[i]
                    yield ev("response.output_item.added", output_index=idx,
                             item={"id": st["fc_id"], "type": "function_call",
                                   "status": "in_progress", "name": st["name"],
                                   "arguments": "", "call_id": st["call_id"]})
                    if not fn.get("arguments"):
                        continue
                st = calls[i]
                frag = fn.get("arguments") or ""
                if not frag:
                    continue
                st["args"] += frag
                yield ev("response.function_call_arguments.delta",
                         item_id=st["fc_id"], output_index=st["index"],
                         delta=frag)

    @app.get("/health")
    async def health():
        rev = {v: k for k, v in DRAFTERS.items()}
        live, took, why = await engine.probe_live()
        body = {"status": "ok" if live else "engine_unresponsive",
                "model": MODEL_ID,
                "endpoints": sorted(
                    r.path for r in app.routes
                    if getattr(r, "path", "").startswith("/v1/")),
                "cache_counters": "/cache",
                "metrics": "/metrics",
                "context": engine.info.get("ctx") or ctx,
                "capability_probe": "ok" if engine.info.get("probed")
                                    else "failed",
                "rope_scaling": ({"type": "yarn",
                                  "factor": engine.info.get("rope_factor"),
                                  "original_context": 262144}
                                 if (engine.info.get("rope_factor") or 1.0) > 1.0
                                 else None),
                "indexer_budget": engine.info.get("indexer_budget", 2048),
                "vision": vision_status(),
                "version": version_status(),
                "modes": modes_status(),
                "checkpoint_format": engine.info.get("checkpoint_format", "hgn"),
                "chat_template": ({k: template[k] for k in
                                   ("path", "sha256", "probe", "thinking_control")}
                                  if template else None),
                "engine": {"responds": live, "probe_s": took,
                           **({"detail": why} if why else {})},
                "busy": engine.slots.locked(),
                "slots": engine.n_slots,
                "slot_ctx": engine.info.get("slot_ctx") or 0,
                "kv_pool_positions": engine.info.get("kv_pool") or 0,
                "in_flight": len(engine.inflight),
                "busy_for_s": (round(time.monotonic() - engine.busy_since, 1)
                               if engine.busy_since and engine.inflight
                               else 0),
                "queued": engine.waiting,
                **({"admission": {"reserve": min(engine.reserve, engine.n_slots - 1),
                                  "priority_in_flight": engine.priority_inflight,
                                  "waiting_for_reserve": engine.waiting_reserve}}
                   if engine.bg is not None else {}),
                "decode": ("greedy at temperature 0 (the default); sampled "
                           "at temperature > 0" if engine.info.get("sampling")
                           else "greedy"),
                "drafters_available": [
                    n for n in DRAFTERS
                    if n not in DRAFTER_CAP
                    or engine.info.get(DRAFTER_CAP[n])],
                "drafter_default": rev.get(engine.info.get("default", 0),
                                           "serial"),
                "prompt_lookup": (
                    {"ngram": engine.info.get("pld_n"),
                     "chain": engine.info.get("pld_k"),
                     "applies_to": "greedy requests with the mtp drafter"}
                    if engine.info.get("pld_n") else "off"),
                "shortlist_draft_head": bool(engine.info.get("draft_head")),
                "drafter_weights_loaded":
                    bool(engine.info.get("drafter_weights")),
                "prompt_cache": {
                    "enabled": bool(engine.info.get("cache_mb")),
                    "cap_mb": engine.info.get("cache_mb", 0),
                    "mode": engine.info.get("cache_mode", 1),
                    "snapshot_align":
                        engine.info.get("cache_align", 0)
                        if engine.info.get("cache_mode", 1) == 1
                        else (engine.info.get("cache_align", 0)
                              if 0 < engine.info.get("cache_align", 0) <= 256 else 0),
                    "bitwise_identical_to_cold":
                        bool(engine.info.get("cache_mb"))
                        and engine.info.get("cache_mode", 1) == 1,
                },
                "composable_context": (
                    {"enabled": True,
                     "unit_floor_tokens": engine.info.get("cc_floor"),
                     "seam_tokens": engine.info.get("cc_seam"),
                     "store_bytes": engine.info.get("cc_bytes"),
                     "fingerprint": engine.info.get("cc_fingerprint"),
                     "bitwise_identical_to_prefill": False,
                     "note": "opt-in; composed context is ~one quantization "
                             "step from a full prefill, not the exact prompt "
                             "cache"}
                    if engine.info.get("composable") else {"enabled": False}),
                "tool_calls": {
                    "wire_format": "qwen-xml (<function=>/<parameter=>)",
                    "streaming": True,
                    "tool_choice": ["auto", "none", "required",
                                    "{type: function, function: {name}}"],
                    "forced_call_is_a_prefill": True,
                    "required_is_exact": "named tool_choice, or one tool",
                    "parallel_tool_calls": True,
                    "constrained_decoding": False,
                    "forced_call_disables_thinking": True,
                    "argument_types_need_schema": True,
                },
                "supported": ["reasoning_effort", "enable_thinking",
                              "preserve_thinking", "max_thinking_tokens",
                              "thinking_budget_tokens", "thinking_budget",
                              "thinking_token_budget", "reasoning", "thinking",
                              "tools", "tool_choice",
                              "parallel_tool_calls", "stop",
                              "max_tokens", "max_completion_tokens",
                              "max_output_tokens", "stream", "stream_options"],
                "token_budget_aliases": ["max_tokens", "max_completion_tokens",
                                         "max_output_tokens"],
                "max_tokens_default": ChatReq.model_fields["max_tokens"].default,
                "reasoning_effort_default": DEFAULTS["reasoning_effort"]
                                            or "xhigh",
                "reasoning_effort_values": list(EFFORT_VALUES),
                "token_budget_covers_reasoning": True,
                "max_thinking_tokens_default": DEFAULTS["max_thinking_tokens"],
                "thinking_answer_room": (DEFAULTS["thinking_answer_room"]
                                         if DEFAULTS["thinking_answer_room"] is not None
                                         else "max(1024, 15% of max_tokens)"),
                "end_of_turn_guard": EOS_GUARD,
                "thinking_control_aliases": ["thinking_budget_tokens", "thinking_budget",
                                             "thinking_token_budget",
                                             "reasoning.{enabled,effort,max_tokens}",
                                             "thinking.{type,budget_tokens}"],
                "server_defaults": defaults_summary(),
                "server_defaults_rule": (
                    "a field the request sends always wins; a default fills "
                    "only a field the request omits. A request that sends "
                    "temperature 0 decodes greedy and takes no sampling "
                    "default"),
                "chat_template_kwargs": list(TEMPLATE_KWARGS),
                "error_format": "openai",
                "accepted_but_ignored": list(SAMPLING_FIELDS),
                "not_implemented": (
                    [] if engine.info.get("grammar")
                    else ["response_format", "text.format"]),
                "structured_output": {
                    "enabled": bool(engine.info.get("grammar")),
                    "reason": engine.info.get("grammar_error") if not engine.info.get("grammar") else None,
                    "response_format": ["json_schema", "json_object"],
                    "text.format": ["json_schema", "json_object"],
                    "routes": ["/v1/chat/completions", "/v1/completions", "/v1/responses",
                               "/v1/messages"],
                    "sampling": "greedy only (temperature 0); sampled + schema is a 400",
                    "images": "a schema with images is a 400",
                    "thinking": "the reasoning block is unconstrained; the "
                                "final text is constrained after </think>",
                    "tools": "a request with tools may open a <tool_call> "
                             "instead of the JSON (the schema binds the "
                             "final text, not a call)",
                    "keys": "object keys are emitted in schema order",
                    "enforced": ["type", "properties", "required",
                                 "additionalProperties", "items", "minItems",
                                 "maxItems", "minLength", "maxLength", "enum",
                                 "const", "anyOf", "oneOf (as anyOf)",
                                 "type: [..., \"null\"]", "$ref", "$defs"],
                    "accepted_not_enforced": ["minimum", "maximum",
                                              "exclusiveMinimum",
                                              "exclusiveMaximum", "multipleOf",
                                              "title", "description",
                                              "default", "examples"],
                    "refused": ["pattern", "format", "allOf", "not", "if",
                                "then", "else", "patternProperties",
                                "dependentRequired", "dependentSchemas",
                                "uniqueItems", "contains", "propertyNames",
                                "unevaluatedProperties", "unevaluatedItems",
                                "minProperties", "maxProperties",
                                "prefixItems"],
                },
                "sampling": ({
                    "implemented": ["temperature", "top_p", "top_k", "min_p",
                                    "seed", "presence_penalty",
                                    "frequency_penalty", "repetition_penalty",
                                    "logit_bias",
                                    "logprobs", "top_logprobs"],
                    "decode": (
                        ("temperature absent samples at the server's "
                         "default of %g (HALOGEN_TEMPERATURE); temperature "
                         "0 is greedy: " % DEFAULTS["temperature"])
                        if DEFAULTS["temperature"] else
                        "temperature 0 (the default) is greedy: ")
                        + "a forward pass and an argmax, byte-identical to "
                          "serial greedy decode. temperature > 0 samples "
                          "from the filtered target on the same drafter "
                          "the request would otherwise get; with the MTP "
                          "drafter the speculative accept/reject rule "
                          "emits exactly the target distribution",
                    "filters": "top_k, top_p and min_p compose as an "
                               "intersection (the HF/vLLM convention); "
                               "top_p >= 1 and top_k = 0 disable a filter. "
                               "A sampled request that omits top_k or top_p "
                               "takes the operator's default, else the "
                               "model's (filter_defaults)",
                    "filter_defaults": dict(MODEL_FILTERS,
                                            source=MODEL_FILTERS_SOURCE),
                    "seed": "reproduces a request on the same drafter; a "
                            "sampled MTP run and a sampled serial run agree "
                            "in distribution, not token for token",
                    "penalties": "presence and frequency count GENERATED "
                                 "tokens only; repetition_penalty scales "
                                 "every id in the prompt or the output "
                                 "(vLLM's rule); applied on both drafters",
                    "logprobs": "a sampled request carries the chosen "
                                "token's logprob on every token; at "
                                "temperature 0, and with top_logprobs (up "
                                "to 20) at any temperature, the first "
                                "generated token only, so those need "
                                "max_tokens 1",
                    "not_implemented": ["logprobs with stream=true",
                                        "logprobs past the first token at "
                                        "temperature 0", "n > 1"],
                } if engine.info.get("sampling") else {
                    "implemented": [],
                    "decode": "greedy only: a forward pass and an argmax",
                    "temperature_gt_0": "NOT IMPLEMENTED by this engine "
                                        "build; rejected with 400 rather "
                                        "than silently served greedy",
                }),
                "max_tokens_cap": max_cap,
                "max_tokens_over_cap": "400",
                "reasoning_effort_values": sorted(set(EFFORT_VALUES))}
        if npu is not None:
            body["npu"] = await npu.health()
        if RC is not None:
            body["request_records"] = RC.health()
        if live:
            return body
        return JSONResponse(status_code=503, content=body)

    @app.get("/metrics")
    async def metrics():
        from fastapi.responses import PlainTextResponse
        kv_pool = int(engine.info.get("kv_pool") or 0) or (
            int(engine.info.get("kv_slots", 1)) * int(engine.info.get("slot_ctx") or 0))
        body = engine.metrics.render(len(engine.inflight), engine.waiting,
                                     sum(engine.reserved.values()), kv_pool,
                                     engine.pool_used,
                                     admission=((min(engine.reserve, engine.n_slots - 1),
                                                 engine.priority_inflight,
                                                 engine.waiting_reserve)
                                                if engine.bg is not None else None))
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/cache")
    async def cache():
        st = await engine.cache_stats()
        if st is None:
            raise HTTPException(501, "this engine reports no prompt cache")
        total = st["hits"] + st["misses"]
        st["hit_rate"] = round(st["hits"] / total, 4) if total else None
        m = engine.metrics
        seen = m.prompt_tokens + m.cache_tokens
        st["token_hit_rate"] = round(m.cache_tokens / seen, 4) if seen else None
        if "pool" in st:
            u, p_ = st["pool"].get("used"), st["pool"].get("positions")
            st["pool"]["usage_ratio"] = round(u / p_, 4) if p_ else None
        if engine.info.get("cache_mode", 1) >= 2:
            st["snapshot_places"] = (["system_end"]
                                     + (["last_user_start"] if SNAP3_ON else [])
                                     + ["history_end"])
        return st

    @app.get("/v1/models")
    async def models():
        window = engine.info.get("ctx") or ctx
        body = {"object": "list", "has_more": False,
                "first_id": MODEL_ID, "last_id": MODEL_ID,
                "data": [{"id": MODEL_ID, "object": "model", "type": "model",
                          "display_name": MODEL_ID,
                          "created_at": "1970-01-01T00:00:00Z",
                          "owned_by": "halogen", "created": 0,
                          "max_model_len": window,
                          "context_length": window,
                          "meta": {"n_ctx_train": window, "n_ctx": window},
                          "max_tokens_cap": max_cap,
                          "max_tokens_default": ChatReq.model_fields["max_tokens"].default}]}
        if npu is not None:
            body["data"] += npu.model_entries()
            body["last_id"] = body["data"][-1]["id"]
        return body

    def render_prompt(msgs, kw, continue_final=False, mark=None):
        if continue_final:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False, **kw)
            cut = text.rstrip()
            if not cut.endswith(EOM):
                raise HTTPException(
                    400, "continue_final_message: this chat template does not "
                         "end the final message with " + EOM + ", so there is "
                         "no place to resume from")
            text = cut[:-len(EOM)]
            stable = None
        else:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True, **kw)
            stable = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=False, **kw)
        mark_at = -1
        if mark and mark in text:
            mark_at = text.find(mark)
            text = text.replace(mark, "", 1)
            if stable is not None:
                stable = stable.replace(mark, "", 1)
            plain = [dict(m) for m in msgs]
            for m in plain:
                c = m.get("content")
                if isinstance(c, str) and mark in c:
                    m["content"] = c.replace(mark, "", 1)
                elif isinstance(c, list):
                    m["content"] = [dict(p, text=p["text"].replace(mark, "", 1))
                                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                                    else p for p in c]
            if text != tok.apply_chat_template(plain, tokenize=False,
                                               add_generation_prompt=True, **kw):
                mark_at = -1
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])

        def boundary_at(cut):
            if not (0 < cut < len(text)):
                return 0
            for i, (a, b) in enumerate(enc["offset_mapping"]):
                if b == cut:
                    return i + 1
                if b > cut:
                    break
            return 0
        seg = []
        c = text.find("<|im_start|>", 1)
        while c > 0:
            b = boundary_at(c)
            if b > 0 and (not seg or b > seg[-1]):
                seg.append(b)
            c = text.find("<|im_start|>", c + 1)
        snap = (boundary_at(len(stable))
                if stable is not None and text.startswith(stable) else 0)
        snap2 = 0
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            cut = text.find("<|im_start|>user")
            if cut > 0:
                snap2 = boundary_at(cut)
            if snap and snap2 >= snap:
                snap2 = 0
        snap3 = 0
        if (SNAP3_ON and not continue_final and msgs
                and isinstance(msgs[-1], dict)
                and msgs[-1].get("role") == "user"):
            first = text.find("<|im_start|>user")
            cut = text.rfind("<|im_start|>user")
            if first > 0 and cut > first:
                snap3 = boundary_at(cut)
            if snap3 and (snap3 <= snap2 or (snap and snap3 >= snap)):
                snap3 = 0
        if SNAP3_ON and mark_at > 0:
            cut = mark_at
            while cut > 0 and text[cut - 1].isspace():
                cut -= 1
            k = boundary_at(cut)
            if k and k > snap2 and (not snap or k < snap):
                snap3 = k
        return ids, snap, snap2, seg, snap3

    def modes_status():
        m = (os.environ.get("HALOGEN_IMAGE_MODES") or "").split()
        return m or None

    def version_status():
        eng = engine.info.get("version") or "unknown"
        known = API_VERSION != "unknown" and eng != "unknown"
        return {"api": API_VERSION, "engine": eng,
                "match": (API_VERSION == eng) if known else None}

    def vision_status():
        why = []
        if not engine.info.get("vision"):
            why.append("the engine was started without a vision tower "
                       "(HALOGEN_VISION_TOWER)")
        if IMAGE_TOKEN_ID is None:
            why.append("this tokenizer has no image placeholder token")
        if not pillow_ready():
            why.append("Pillow is not installed for the front-end")
        st = {"enabled": not why,
              "endpoints": ["/v1/chat/completions", "/v1/messages",
                            "/v1/responses"],
              "input": "data: URL or bare base64 in an image content part; "
                       "http(s) URLs are refused",
              "max_pixels": VIS_MAX_PIXELS,
              "size_multiple": VIS_UNIT}
        if why:
            st["disabled_because"] = why
        return st

    def collect_images(msgs):
        try:
            imgs = vis_collect(msgs)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if imgs:
            st = vision_status()
            if not st["enabled"]:
                raise HTTPException(400, "this server does not accept images: "
                                    + "; ".join(st["disabled_because"]))
        return imgs

    def continued_text(msg):
        c = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(p.get("text", "") for p in c
                           if isinstance(p, dict) and p.get("type") == "text")
        return ""

    def open_call_head(partial):
        i = partial.rfind(TC_OPEN)
        if i >= 0 and TC_CLOSE not in partial[i:]:
            return partial[i:]
        for k in range(len(TC_OPEN) - 1, 0, -1):
            if partial.endswith(TC_OPEN[:k]):
                return partial[-k:]
        return ""

    def render_with_images(msgs, kw, imgs, continue_final=False, mark=None):
        try:
            ids, snap, snap2, seg, snap3 = render_prompt(msgs, kw, continue_final,
                                                          mark)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"chat template rejected the request: "
                                     f"{e}")
        ids0 = ids
        ids, snap, snap2, _ = textify_pad_mentions(ids0, snap, snap2, len(imgs))
        if snap3:
            snap3 = textify_pad_mentions(ids0, snap3, 0, len(imgs))[1]
        if not imgs:
            return ids, snap, snap2, None, seg, snap3
        ids1 = ids
        ids, snap, snap2, placed = expand_image_pads(ids1, snap, snap2, imgs)
        if snap3:
            snap3 = expand_image_pads(ids1, snap3, 0, imgs)[1]
        return ids, snap, snap2, [(f, h, w, imgs[i][2])
                                  for i, (f, h, w) in enumerate(placed)], [], snap3

    async def serve(ids, max_tokens, stops, stream, chat, prefix,
                    thinking=True, drafter=None, tools=None, parallel=True,
                    sample=None, penalty="",
                    pre="", forced=False, include_usage=False, snap=0,
                    snap2=0, wire="chat", images=None, want_summary=False,
                    schema=None, think_budget=None, seg=None, snap3=0,
                    http_request=None, show_thinking=True, rec=None):
        after = THINK_END_ID if (schema and thinking and chat) else None
        escape = [TOOL_CALL_ID] if (schema and tools and TOOL_CALL_ID is not None) else ()
        think = None
        think_src = None
        if think_budget and thinking and chat and THINK_END_ID is not None:
            think = (int(think_budget), THINK_END_ID, THINK_CLOSE_IDS)
            think_src = "max_thinking_tokens"
        if thinking and chat and THINK_END_ID is not None:
            room = answer_room(max_tokens)
            if room > 0:
                cap = max(1, int(max_tokens) - room)
                if think is None or think[0] > cap:
                    think = (cap, THINK_END_ID, THINK_CLOSE_IDS)
                    think_src = "answer_room"
        think_on = bool(thinking) if chat else None
        guard = guard_for(ids, bool(tools)) if chat else None
        want = int(max_tokens)
        if want > max_cap:
            raise HTTPException(
                400, f"max_tokens {want} exceeds this server's cap of "
                     f"{max_cap}. The cap is server policy, not a model "
                     f"limit: halogen is batch-1, so one long request holds "
                     f"the GPU for its whole run and every other client "
                     f"queues behind it. Raise it with --max-tokens-cap; "
                     f"/health reports the live value as max_tokens_cap.")
        limit = engine.info.get("ctx") or ctx
        room = limit - len(ids)
        if room <= 0:
            raise HTTPException(
                400, f"the prompt does not fit: it is {len(ids)} tokens and "
                     f"the context is {limit}, {-room} tokens over. Shorten "
                     f"the prompt, or raise the context (HALOGEN_CTX, and "
                     f"HALOGEN_ROPE_YARN for more than the native 262144).")
        if want > room:
            raise HTTPException(
                400, f"max_tokens {want} does not fit: prompt is {len(ids)} "
                     f"tokens and the context is {limit}, leaving room for "
                     f"{room}.")
        max_tokens = max(1, want)

        hinted = priority_hinted(http_request)
        t_enter = time.monotonic()
        bg = await engine.admit(hinted, queue_timeout)
        slot_key = uuid.uuid4().hex
        if bg is not None:
            engine.bg_of[slot_key] = bg
        if hinted:
            engine.priority_inflight += 1
            engine.hinted_keys.add(slot_key)
        engine.inflight[slot_key] = time.monotonic()
        engine.reserved[slot_key] = len(ids) + int(max_tokens)
        engine.busy_since = min(engine.inflight.values())
        cid = (f"resp_{uuid.uuid4().hex[:24]}" if wire == "responses"
               else f"msg_{uuid.uuid4().hex[:24]}" if wire == "messages"
               else f"{prefix}-{uuid.uuid4().hex[:24]}")
        created = int(time.time())
        split = ThinkSplit(thinking)
        if rec is not None:
            rec.admitted(cid, (time.monotonic() - t_enter) * 1000.0, {
                "n_prompt": len(ids), "n_tools": len(tools or ()),
                "n_images": len(images or ()), "think": bool(think_on),
                "sampled": bool(sample), "schema": bool(schema),
                "priority": bool(hinted), "forced": bool(forced)})
        json_only = bool(schema) and bool(tools) and SCHEMA_PREFACE
        tstream = ToolStream(tools, json_only=json_only) if tools else None
        head = ""
        if guard and guard[0] and pre and not forced:
            head, pre = pre, ""
        elif pre and not forced:
            pre = open_call_head(pre)
        block_calls = bool(guard and guard[0] and tools and thinking and not forced)

        def rec_end(finish, d, call_ids):
            if rec is None:
                return
            try:
                if chat:
                    r, c = split.parts()
                else:
                    r, c = "", split.full
                eng = None
                if d and "n_prompt" in d:
                    try:
                        eng = timings(d)
                    except Exception:
                        eng = None
                rec.finish(finish, d, eng, r or "", (pre + (c or "")) if chat else (c or ""),
                           call_ids, tools, json_only)
            except Exception:
                pass

        hold = {}

        async def tap(g):
            try:
                async for delta, d in g:
                    if d is not None:
                        hold["d"] = d
                    yield delta, d
            finally:
                await g.aclose()
        if stream:
            n_ev = 0
            n_ka = 0
            finished = False
            t_body = time.perf_counter()
            t_last = t_body
            gap_max = 0.0

            async def body():
                nonlocal n_ev, n_ka, finished, t_last, gap_max
                try:
                    gen = run(ids, max_tokens, stops, drafter, sample,
                              penalty, snap, snap2, images, schema,
                              after, escape, think, seg, snap3=snap3,
                              think_on=think_on, think_src=think_src,
                              guard=guard)
                    if block_calls:
                        gen = call_in_block(gen, split, tools, head)
                    gen = guarded(gen)
                    if rec is not None:
                        gen = tap(gen)
                    stream_iter = (
                        sse_responses(gen, cid, created, MODEL_ID, split,
                                      tstream, pre, parallel, max_tokens,
                                      want_summary)
                        if wire == "responses" else
                        sse_messages(gen, cid, MODEL_ID, split, tstream, pre,
                                     parallel, len(ids), show_thinking)
                        if wire == "messages" else
                        sse(gen, cid, created, chat, split, tstream, pre,
                            parallel, include_usage))
                    async for ev in keepalive(stream_iter, SSE_KEEPALIVE_S):
                        if ev.startswith(":"):
                            n_ka += 1
                            yield ('event: ping\ndata: {"type": "ping"}\n\n'
                                   if wire == "messages" else ev)
                            continue
                        now = time.perf_counter()
                        gap_max = max(gap_max, now - t_last)
                        t_last = now
                        n_ev += 1
                        yield ev
                    finished = True
                finally:
                    if not finished:
                        now = time.perf_counter()
                        print("serve_api: client disconnected mid-stream, %s, "
                              "%d SSE frames and %d chars sent in %.1fs, "
                              "largest gap between frames %.1fs, last frame "
                              "%.1fs before the hangup, %d keepalives "
                              "(the engine is being stopped; this is the "
                              "client hanging up, not a server error)"
                              % (cid, n_ev, len(split.full), now - t_body,
                                 max(gap_max, now - t_last), now - t_last,
                                 n_ka),
                              flush=True)
                    try:
                        await asyncio.wait_for(engine.abort(),
                                               timeout=ABORT_DRAIN_S + 5)
                    except Exception:
                        await engine.close()
                    finally:
                        engine.release_admission(slot_key); engine.inflight.pop(slot_key, None); engine.reserved.pop(slot_key, None)
                        engine.busy_since = (min(engine.inflight.values())
                                             if engine.inflight else None)
                        engine.slots.release()
                    if rec is not None:
                        d_ = hold.get("d")
                        fin_ = ("disconnect" if not finished else
                                (d_ or {}).get("reason", "stop"))
                        rec_end(fin_, d_, [c["id"] for c in tstream.calls if c]
                                if tstream is not None else [])
            return StreamingResponse(body(), media_type="text/event-stream")
        try:
            text, done = "", None
            gen = run(ids, max_tokens, stops, drafter, sample, penalty, snap,
                      snap2, images, schema, after, escape, think, seg,
                      snap3=snap3, think_on=think_on, think_src=think_src,
                      guard=guard)
            if block_calls:
                gen = call_in_block(gen, split, tools, head)
            gone, n_tok = False, 0
            watch = {}
            try:
                async for delta, d in until_gone(gen, http_request, watch):
                    if d is not None:
                        done = d
                        break
                    text += delta
                    n_tok += 1
                    if http_request is not None and await http_request.is_disconnected():
                        gone = True
                        break
            finally:
                await gen.aclose()
            gone = gone or bool(watch.get("gone"))
            if gone:
                print("serve_api: client disconnected mid-request (non-streaming), "
                      "%d token(s) generated and dropped; the generation is cancelled"
                      % n_tok, flush=True)
                split.full = text
                rec_end("disconnect", done, [])
                return Response(status_code=499)
        finally:
            try:
                await asyncio.wait_for(engine.abort(),
                                       timeout=ABORT_DRAIN_S + 5)
            except Exception:
                await engine.close()
            finally:
                engine.release_admission(slot_key); engine.inflight.pop(slot_key, None); engine.reserved.pop(slot_key, None)
                engine.busy_since = (min(engine.inflight.values())
                                     if engine.inflight else None)
                engine.slots.release()
        fin = {"stop": "stop", "cancel": "stop",
               "length": "length"}.get(done["reason"], "stop")
        if wire == "messages":
            split.full = text
            reasoning, content = split.parts()
            tool_calls = []
            if tools:
                content, tool_calls, _ = split_tool_calls(pre + content, tools, json_only=json_only)
                if not parallel:
                    tool_calls = tool_calls[:1]
            if forced and not tool_calls:
                split.full = text
                rec_end("error", done, [])
                raise HTTPException(502, "tool_choice required a call and the "
                                         "model did not name a tool: retry, or "
                                         "name the tool in tool_choice")
            blocks = []
            if reasoning:
                blocks.append({"type": "thinking",
                               "thinking": reasoning if show_thinking else "",
                               "signature": think_signature(reasoning)})
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in tool_calls:
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": tc["function"]["name"],
                               "input": tool_input(tc["function"]["arguments"])})
            done["n_reasoning"] = reasoning_tokens(done, split.thinking)
            rec_end("tool_calls" if tool_calls else fin, done,
                    [tc["id"] for tc in tool_calls])
            sr, ss = messages_stop(done, bool(tool_calls))
            return {"id": cid, "type": "message", "role": "assistant",
                    "model": MODEL_ID, "content": blocks, "stop_reason": sr,
                    "stop_sequence": ss,
                    "usage": messages_usage(done, len(ids)),
                    "timings": timings(done)}
        if wire == "responses":
            split.full = text
            reasoning, content = split.parts()
            tool_calls = []
            if tools:
                content, tool_calls, _ = split_tool_calls(pre + content, tools, json_only=json_only)
                if not parallel:
                    tool_calls = tool_calls[:1]
            out = []
            if reasoning:
                out.append({"id": "rs_" + cid.split("_", 1)[-1],
                            "type": "reasoning", "status": "completed",
                            "summary": ([{"type": "summary_text",
                                          "text": reasoning}]
                                        if want_summary else []),
                            "content": [{"type": "reasoning_text",
                                         "text": reasoning}],
                            "encrypted_content": None})
            if content:
                out.append({"id": "msg_" + cid.split("_", 1)[-1],
                            "type": "message", "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text",
                                         "text": content,
                                         "annotations": []}]})
            for tc in tool_calls:
                out.append({"id": f"fc_{uuid.uuid4().hex[:24]}",
                            "type": "function_call", "status": "completed",
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                            "call_id": tc["id"]})
            done["n_reasoning"] = reasoning_tokens(done, split.thinking)
            rec_end("tool_calls" if tool_calls else fin, done,
                    [tc["id"] for tc in tool_calls])
            body = {"id": cid, "object": "response", "created_at": created,
                    "status": "incomplete" if fin == "length" else "completed",
                    "model": MODEL_ID, "output": out,
                    "parallel_tool_calls": parallel,
                    "max_output_tokens": max_tokens,
                    "usage": responses_usage(done, len(ids)),
                    "timings": timings(done)}
            if fin == "length":
                body["incomplete_details"] = {"reason": "max_output_tokens"}
            return body
        if chat:
            split.full = text
            reasoning, content = split.parts()
            tool_calls = []
            if tools:
                content, tool_calls, _ = split_tool_calls(pre + content, tools, json_only=json_only)
                if not parallel:
                    tool_calls = tool_calls[:1]
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if forced and not tool_calls:
                rec_end("error", done, [])
                raise HTTPException(502, "tool_choice required a call and the "
                                         "model did not name a function: "
                                         "retry, or name the function in "
                                         "tool_choice")
            if tool_calls:
                msg["tool_calls"] = tool_calls
                if fin != "length":
                    fin = "tool_calls"
            choice = {"index": 0, "finish_reason": fin, "message": msg}
        else:
            choice = {"index": 0, "finish_reason": fin, "text": text}
        if done and done.get("logprobs"):
            choice["logprobs"] = {"content": [
                logprob_entry(tok, t, v) for t, v in done["logprobs"]]}
        done["n_reasoning"] = reasoning_tokens(done, split.thinking if chat else None)
        if not chat:
            split.full = text
        rec_end(choice.get("finish_reason", fin), done,
                [tc["id"] for tc in (choice.get("message") or {}).get("tool_calls", ())])
        return {"id": cid, "created": created, "model": MODEL_ID,
                "object": "chat.completion" if chat else "text_completion",
                "choices": [choice], "usage": usage(done, len(ids)),
                "timings": timings(done)}

    @app.post("/v1/completions")
    async def completions(req: CompletionReq, http_request: Request):
        if npu is not None and npu.owns(req.model):
            return oai_error(400, npu.wrong_route(req.model, "/v1/completions"))
        ids = tok(req.prompt, add_special_tokens=False)["input_ids"]
        ids, _, _, _ = textify_pad_mentions(list(ids))
        check_sampling(req)
        return await serve(ids, req.max_tokens, stop_list(req.stop),
                           req.stream, False, "cmpl",
                           http_request=http_request,
                           rec=rec_begin("completions", http_request, [], None, 1,
                                         req.model, req.stream),
                           drafter=drafter_for(req),
                           sample=sample_spec(req),
                           penalty=penalty_spec(req),
                           include_usage=wants_usage(req),
                           schema=schema_for(req))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatReq, http_request: Request):
        if npu is not None and npu.owns(req.model):
            return await npu.chat(req, http_request)
        check_sampling(req)
        for f in SAMPLING_FIELDS:
            if getattr(req, f) is not None and f not in _warned_sampling:
                _warned_sampling.add(f)
                print(f"serve_api: '{f}' accepted but IGNORED "
                      f"by this endpoint", flush=True)
        mode, forced = tool_choice_mode(req)
        pre = force_prefill(mode, forced, req.tools)
        thinking = thinking_on(req)
        kw = chat_kwargs(req, mode)
        if pre:
            kw["enable_thinking"] = False
            thinking = False
        try:
            msgs = normalize_messages(req.messages)
        except ValueError as e:
            raise HTTPException(400, str(e))
        imgs = collect_images(msgs)
        cont = bool(req.continue_final_message)
        if cont and pre:
            raise HTTPException(400, "continue_final_message cannot be combined "
                                     "with a forced tool choice: one resumes the "
                                     "assistant turn already in the history, the "
                                     "other starts a new call")
        ids, snap, snap2, wire_imgs, seg, snap3 = render_with_images(
            msgs, kw, imgs, cont)
        if pre:
            ids += tok(pre, add_special_tokens=False)["input_ids"]
        elif cont:
            pre = continued_text(msgs[-1])
            st = prompt_state(ids)
            if thinking and st is not None and not st[0]:
                thinking = False
        md = req.metadata if isinstance(req.metadata, dict) else {}
        return await serve(ids, req.max_tokens, stop_list(req.stop),
                           req.stream, True, "chatcmpl",
                           http_request=http_request,
                           rec=rec_begin("chat", http_request,
                                         [("prompt_cache_key", req.prompt_cache_key),
                                          ("safety_identifier", req.safety_identifier),
                                          ("metadata.session_id", md.get("session_id")),
                                          ("metadata.user_id", md.get("user_id")),
                                          ("user", req.user)],
                                         ("chat", req.messages), len(req.messages),
                                         req.model, req.stream, req.reasoning_effort,
                                         mode),
                           thinking=thinking,
                           think_budget=server_default(req, "max_thinking_tokens"),
                           drafter=drafter_for(req),
                           sample=sample_spec(req),
                           penalty=penalty_spec(req),
                           tools=(req.tools if mode != "none" else None),
                           parallel=req.parallel_tool_calls is not False,
                           pre=pre, forced=bool(pre) and not cont,
                           include_usage=wants_usage(req), snap=snap,
                           snap2=snap2, images=wire_imgs,
                           schema=schema_for(req), seg=seg, snap3=snap3)

    @app.post("/v1/responses")
    async def responses(req: ResponsesReq, http_request: Request):
        if npu is not None and npu.owns(req.model):
            return oai_error(400, npu.wrong_route(req.model, "/v1/responses"))
        chat_tools, dropped = responses_tools(req.tools)
        if dropped:
            key = ",".join(sorted(set(dropped)))
            if key not in _warned_sampling:
                _warned_sampling.add(key)
                print(f"serve_api: /v1/responses ignored {len(dropped)} "
                      f"non-function tool(s) of type: {key}", flush=True)
        tc = req.tool_choice
        if isinstance(tc, dict) and tc.get("type") == "function" \
                and "function" not in tc:
            tc = {"type": "function", "function": {"name": tc.get("name")}}
        shim = ChatReq(model=req.model, messages=[{"role": "user",
                                                   "content": ""}],
                       max_tokens=req.max_output_tokens or CHAT_MAX_TOKENS,
                       stream=req.stream, tools=chat_tools, tool_choice=tc,
                       parallel_tool_calls=req.parallel_tool_calls,
                       reasoning=req.reasoning if isinstance(req.reasoning, dict) else None,
                       drafter=req.drafter,
                       max_thinking_tokens=getattr(req, "max_thinking_tokens", None),
                       thinking_budget_tokens=getattr(req, "thinking_budget_tokens", None),
                       thinking_budget=getattr(req, "thinking_budget", None),
                       thinking_token_budget=getattr(req, "thinking_token_budget", None),
                       thinking=getattr(req, "thinking", None),
                       temperature=req.temperature, top_p=req.top_p,
                       response_format=(req.text or {}).get("format"))
        check_sampling(shim)
        mode, forced = tool_choice_mode(shim)
        pre = force_prefill(mode, forced, chat_tools)
        thinking = thinking_on(shim)
        kw = chat_kwargs(shim, mode)
        if pre:
            kw["enable_thinking"] = False
            thinking = False
        try:
            msgs = normalize_messages(
                responses_messages(req.instructions, req.input))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not msgs or all(m.get("role") == "system" for m in msgs):
            raise HTTPException(400, "input is empty: /v1/responses needs at "
                                     "least one non-system turn")
        imgs = collect_images(msgs)
        ids, snap, snap2, wire_imgs, seg, snap3 = render_with_images(msgs, kw, imgs)
        if pre:
            ids += tok(pre, add_special_tokens=False)["input_ids"]
        cm = req.client_metadata if isinstance(req.client_metadata, dict) else {}
        mdr = req.metadata if isinstance(req.metadata, dict) else {}
        return await serve(ids, shim.max_tokens, stop_list(None),
                           req.stream, True, "resp",
                           http_request=http_request,
                           rec=rec_begin("responses", http_request,
                                         [("prompt_cache_key", req.prompt_cache_key),
                                          ("safety_identifier", req.safety_identifier),
                                          ("client_metadata", cm or None),
                                          ("metadata.session_id", mdr.get("session_id")),
                                          ("user", req.user)],
                                         ("responses", req.instructions, req.input,
                                          responses_messages),
                                         len(req.input) if isinstance(req.input, list) else 1,
                                         req.model, req.stream, shim.reasoning_effort,
                                         mode),
                           thinking=thinking,
                           think_budget=server_default(shim, "max_thinking_tokens"),
                           drafter=drafter_for(shim),
                           sample=sample_spec(shim),
                           penalty=penalty_spec(shim),
                           tools=(chat_tools if mode != "none" else None),
                           parallel=req.parallel_tool_calls is not False,
                           pre=pre, forced=bool(pre),
                           snap=snap, snap2=snap2, wire="responses",
                           images=wire_imgs, seg=seg, snap3=snap3,
                           want_summary=bool((req.reasoning or {}).get("summary")
                                             not in (None, "", "none")),
                           schema=schema_for(shim))

    MESSAGES_FIELDS = set(MessagesReq.model_fields)

    def messages_prompt(req):
        for f in sorted(set(req.model_extra or {}) - MESSAGES_FIELDS):
            if ("messages:" + f) not in _warned_sampling:
                _warned_sampling.add("messages:" + f)
                print(f"serve_api: /v1/messages ignored the field '{f}'",
                      flush=True)
        chat_tools, dropped = messages_tools(req.tools)
        if dropped:
            key = "messages-tools:" + ",".join(sorted(set(dropped)))
            if key not in _warned_sampling:
                _warned_sampling.add(key)
                print(f"serve_api: /v1/messages ignored {len(dropped)} "
                      f"Anthropic-defined tool(s) of type: "
                      f"{','.join(sorted(set(dropped)))}", flush=True)
        tc, parallel = messages_tool_choice(req.tool_choice)
        th = req.thinking if isinstance(req.thinking, dict) else {}
        ty = th.get("type")
        enable = None if not th else ty != "disabled"
        budget = th.get("budget_tokens") if ty == "enabled" else None
        oc = req.output_config if isinstance(req.output_config, dict) else {}
        fmt = oc.get("format")
        rf = ({"type": "json_schema", "json_schema": {
                  "name": "output", "schema": fmt.get("schema") or {}}}
              if isinstance(fmt, dict) and fmt.get("type") == "json_schema"
              else None)
        shim = ChatReq(model=req.model, messages=[{"role": "user",
                                                   "content": ""}],
                       max_tokens=req.max_tokens, stream=req.stream,
                       tools=chat_tools, tool_choice=tc,
                       parallel_tool_calls=parallel,
                       reasoning_effort=oc.get("effort"),
                       enable_thinking=enable, max_thinking_tokens=budget,
                       temperature=req.temperature, top_p=req.top_p,
                       top_k=req.top_k, stop=req.stop_sequences,
                       response_format=rf)
        check_sampling(shim)
        mode, forced = tool_choice_mode(shim)
        pre = force_prefill(mode, forced, chat_tools)
        thinking = thinking_on(shim)
        kw = chat_kwargs(shim, mode)
        if pre:
            kw["enable_thinking"] = False
            thinking = False
        turns = mark_last_cache_point(req.messages, CACHE_MARK)
        try:
            msgs = normalize_messages(messages_to_openai(req.system, turns))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not msgs or all(m.get("role") == "system" for m in msgs):
            raise HTTPException(400, "messages is empty: /v1/messages needs "
                                     "at least one user turn")
        imgs = collect_images(msgs)
        ids, snap, snap2, wire_imgs, seg, snap3 = render_with_images(
            msgs, kw, imgs, mark=CACHE_MARK if turns is not req.messages else None)
        if pre:
            ids += tok(pre, add_special_tokens=False)["input_ids"]
        return (shim, ids, snap, snap2, snap3, wire_imgs, seg, chat_tools,
                mode, pre, thinking, th.get("display") != "omitted")

    @app.post("/v1/messages")
    async def messages(req: MessagesReq, http_request: Request):
        if npu is not None and npu.owns(req.model):
            return anth_error(400, npu.wrong_route(req.model, "/v1/messages"))
        (shim, ids, snap, snap2, snap3, wire_imgs, seg, chat_tools, mode,
         pre, thinking, show) = messages_prompt(req)
        mdm = req.metadata if isinstance(req.metadata, dict) else {}
        return await serve(ids, shim.max_tokens, stop_list(shim.stop),
                           req.stream, True, "msg",
                           http_request=http_request,
                           rec=rec_begin("messages", http_request,
                                         [("metadata.user_id", mdm.get("user_id")),
                                          ("metadata.session_id", mdm.get("session_id"))],
                                         ("messages", req.system, req.messages,
                                          messages_to_openai),
                                         len(req.messages), req.model, req.stream,
                                         shim.reasoning_effort, mode),
                           thinking=thinking,
                           think_budget=server_default(shim, "max_thinking_tokens"),
                           drafter=drafter_for(shim),
                           sample=sample_spec(shim),
                           penalty=penalty_spec(shim),
                           tools=(chat_tools if mode != "none" else None),
                           parallel=shim.parallel_tool_calls is not False,
                           pre=pre, forced=bool(pre),
                           snap=snap, snap2=snap2, wire="messages",
                           images=wire_imgs, seg=seg, snap3=snap3,
                           schema=schema_for(shim), show_thinking=show)

    @app.post("/v1/messages/count_tokens")
    async def messages_count_tokens(http_request: Request):
        try:
            body = await http_request.json()
        except Exception:
            raise HTTPException(400, "the body is not JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "the body must be a JSON object")
        body = dict(body, max_tokens=body.get("max_tokens") or 1,
                    stream=False)
        try:
            req = MessagesReq(**body)
        except Exception as e:
            raise HTTPException(400, str(e)[:300])
        ids = messages_prompt(req)[1]
        return {"input_tokens": len(ids)}

    @app.api_route("/api/hello", methods=["GET", "HEAD"])
    async def api_hello():
        return Response(status_code=200)

    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True,
                    help="HF snapshot dir or model id (tokenizer only)")
    ap.add_argument("--engine", default="127.0.0.1:8730",
                    help="the Flash engine's HOST:PORT, or none: the NPU "
                         "decision models alone (--npu)")
    ap.add_argument("--npu", default=None,
                    help="HOST:PORT of the NPU decision engine (halogen-npu "
                         "listen); a request naming one of its models goes "
                         "there, every other request to the Flash engine")
    ap.add_argument("--npu-tokenizer", default=None, action="append",
                    help="an NPU model's tokenizer directory, repeatable (each "
                         "matched to its model by the sha256 its weight file "
                         "records; with --engine none, --tokenizer is one)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--context", type=int, default=32768,
                    help="FALLBACK only: the engine's INFO reports the real "
                         "context and that is what is enforced. The "
                         "entrypoint passes HALOGEN_CTX here (0.13.1, issue "
                         "#90) so a failed probe cannot shrink the context.")
    ap.add_argument("--queue-timeout", type=float, default=3600,
                    help="seconds a request waits for the batch-1 slot "
                         "before 503")
    ap.add_argument("--max-tokens-cap", type=int, default=65536,
                    help="server-side ceiling on max_tokens (batch-1: one "
                         "long request blocks every other client)")
    ap.add_argument("--check-defaults", action="store_true",
                    help="validate the HALOGEN_* request defaults and exit")
    args = ap.parse_args()

    if DEFAULTS["max_tokens"] and DEFAULTS["max_tokens"] > args.max_tokens_cap:
        raise SystemExit(
            f"serve_api: HALOGEN_MAX_TOKENS_DEFAULT={DEFAULTS['max_tokens']} "
            f"exceeds the cap of {args.max_tokens_cap} "
            f"(HALOGEN_MAX_TOKENS_CAP); every request that omitted a budget "
            f"would be refused. Lower the default or raise the cap.")
    if args.check_defaults:
        d = defaults_summary()
        print("serve_api: request defaults "
              + (", ".join(f"{k}={v}" for k, v in d.items()) if d
                 else "none set (built-in)"), flush=True)
        from transformers import AutoTokenizer
        check_template_or_exit(AutoTokenizer.from_pretrained(args.tokenizer),
                               args.tokenizer)
        return

    if args.engine == "none":
        import npu_api
        return npu_api.serve_only(args, ChatReq, oai_error, KEEPALIVE_S)
    npu = None
    if args.npu:
        import npu_api
        npu = npu_api.Front(args.npu, args.npu_tokenizer, oai_error,
                            args.queue_timeout)

    import uvicorn
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    template = check_template_or_exit(tok, args.tokenizer)
    if vis_resolve_token(tok) is None:
        print("serve_api: no <|image_pad|> in this tokenizer; image requests "
              "will be refused", flush=True)
    host, _, port = args.engine.partition(":")
    engine = Engine(host, int(port))
    app = build_app(tok, engine, args.context, args.max_tokens_cap,
                    args.queue_timeout, template=template, npu=npu)
    if npu is not None:
        npu.add_routes(app, ours=(MODEL_ID,))
        if app.state.records is not None:
            npu.attach_records(app.state.records)

    @app.on_event("startup")
    async def _connect():
        await engine.connect()
        if npu is not None:
            try:
                await npu.start()
            except Exception as e:
                print(f"serve_api: {e}", flush=True)
                os._exit(1)
        print(f"serve_api: engine at {args.engine}, "
              f"listening on {args.host}:{args.port}", flush=True)
        if DEFAULTS["temperature"] and not engine.info.get("sampling"):
            print("serve_api: HALOGEN_TEMPERATURE is set but this engine "
                  "build decodes greedy only; unset it or use a build that "
                  "samples", flush=True)
            os._exit(1)
        d = defaults_summary()
        if d:
            print("serve_api: request defaults "
                  + ", ".join(f"{k}={v}" for k, v in d.items())
                  + " (a field the request sends wins; temperature 0 in a "
                    "request is greedy and takes no sampling default)",
                  flush=True)
        if engine.info.get("sampling"):
            fk = DEFAULTS["top_k"] if DEFAULTS["top_k"] is not None \
                else MODEL_FILTERS["top_k"]
            fp = DEFAULTS["top_p"] if DEFAULTS["top_p"] is not None \
                else MODEL_FILTERS["top_p"]
            print("serve_api: a sampled request that omits top_k / top_p "
                  "gets top_k %d, top_p %g (HALOGEN_TOP_K / HALOGEN_TOP_P "
                  "where set, else the model's %s)"
                  % (fk, fp, MODEL_FILTERS_SOURCE), flush=True)
        eng = engine.info.get("version") or "unknown"
        print(f"serve_api: version {API_VERSION}, engine version {eng}",
              flush=True)
        if eng != API_VERSION:
            print(f"serve_api: WARNING the api ({API_VERSION}) and the engine "
                  f"({eng}) are different releases. Run both containers from "
                  f"the same image tag: what one of them can do, the other "
                  f"may not know how to ask for (images, for one), and the "
                  f"answer may be wrong without any error saying so.",
                  flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                timeout_keep_alive=KEEPALIVE_S)

if __name__ == "__main__":
    main()
