#!/usr/bin/env python3
# halogen-flash-server: source shipped as-is, comments stripped. The README is the documentation.
import json
import math
import re
import uuid

TC_OPEN = "<tool_call>"
TC_CLOSE = "</tool_call>"
PARAM_END = "</parameter>"

_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FUNC_RE = re.compile(r"<function=([^>\n]*)>[ \t]*\n?")
_PARAM_RE = re.compile(r"<parameter=([^>\n]*)>[ \t]*\n?")
_AFTER_END_RE = re.compile(r"\s*(<parameter=|</function>|</tool_call>)")

def new_call_id():
    return "call_" + uuid.uuid4().hex[:24]

def schema_types(tools):
    out = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        props = (fn.get("parameters") or {}).get("properties")
        types = {}
        if isinstance(props, dict):
            for k, v in props.items():
                types[k] = resolve_type(v)
        out[name] = types
    return out

_TYPE_ALIAS = {"str": "string", "text": "string", "varchar": "string",
               "char": "string", "enum": "string", "bool": "boolean",
               "binary": "boolean", "list": "array", "sequence": "array",
               "tuple": "array", "arr": "array", "dict": "object",
               "map": "object", "mapping": "object", "json": "object",
               "none": "null"}

def norm_type(t):
    if not isinstance(t, str):
        return None
    t = t.strip().lower()
    if t in ("string", "integer", "number", "boolean", "array", "object",
             "null"):
        return t
    if t in _TYPE_ALIAS:
        return _TYPE_ALIAS[t]
    if t.startswith(("int", "uint", "long", "short", "unsigned")):
        return "integer"
    if t.startswith(("num", "float", "double", "decimal")):
        return "number"
    return None

NULLABLE_STRING = "string?"

def resolve_type(schema):
    if not isinstance(schema, dict):
        return None
    found, nullable = [], False

    def walk(sch, depth):
        nonlocal nullable
        if not isinstance(sch, dict) or depth > 4:
            found.append(None)
            return
        t = sch.get("type")
        if isinstance(t, list):
            members = t
        elif t is not None:
            members = [t]
        else:
            members = []
        for m in members:
            n = norm_type(m)
            if n == "null":
                nullable = True
            else:
                found.append(n)
        for key in ("anyOf", "oneOf"):
            if isinstance(sch.get(key), list):
                for b in sch[key]:
                    walk(b, depth + 1)
        if not members and not isinstance(sch.get("anyOf"), list) \
                and not isinstance(sch.get("oneOf"), list):
            found.append(None)

    walk(schema, 0)
    kinds = set(found)
    if not kinds:
        return None
    if None in kinds:
        return None
    if len(kinds) == 1:
        t = next(iter(kinds))
    elif "string" in kinds:
        t = "string"
    elif kinds <= {"integer", "number"}:
        t = "number"
    else:
        return None
    return NULLABLE_STRING if t == "string" and nullable else t

def coerce(raw, jtype):
    if jtype == "string":
        return raw
    if jtype == NULLABLE_STRING:
        return None if raw.strip() == "null" else raw
    v = raw.strip()
    if jtype is None:
        try:
            p = json.loads(v)
        except Exception:
            return raw
        return raw if isinstance(p, str) else p
    if v.lower() == "null":
        return None
    if jtype == "boolean" and v.lower() in ("true", "false"):
        return v.lower() == "true"
    if jtype in ("integer", "number"):
        try:
            return int(v)
        except ValueError:
            pass
        try:
            f = float(v)
        except ValueError:
            f = None
        if f is not None:
            if not math.isfinite(f):
                return raw
            if jtype == "number":
                return f
            if f.is_integer():
                return int(f)
    try:
        return json.loads(v)
    except Exception:
        return raw

def _value_end(body, vstart, final):
    nxt = body.find("<parameter=", vstart)
    limit = len(body) if nxt < 0 else nxt
    i, last = body.find(PARAM_END, vstart), None
    while 0 <= i < limit:
        last = i
        if _AFTER_END_RE.match(body[i + len(PARAM_END):]):
            return i, i + len(PARAM_END)
        i = body.find(PARAM_END, i + 1)
    if last is not None and (final or nxt >= 0):
        return last, last + len(PARAM_END)
    if nxt >= 0:
        return nxt, nxt
    if final:
        return len(body), len(body)
    return None, None

def _scan_params(body, final=True):
    out, pos = [], 0
    while True:
        m = _PARAM_RE.search(body, pos)
        if not m:
            return out, None
        end, nxt = _value_end(body, m.end(), final)
        if end is None:
            return out, (m.group(1).strip(), body[m.end():])
        raw = body[m.end():end]
        if raw.endswith("\n"):
            raw = raw[:-1]
        out.append((m.group(1).strip(), raw))
        pos = nxt

def _parse_params(body, final=True):
    return _scan_params(body, final)[0]

def _stream_safe(sofar):
    j = sofar.rfind(PARAM_END)
    if j >= 0:
        sofar = sofar[:j]
    else:
        k = sofar.rfind("<")
        if k >= 0 and (PARAM_END.startswith(sofar[k:])
                       or "<parameter=".startswith(sofar[k:])):
            sofar = sofar[:k]
    return sofar.rstrip()

def _parse_block(block, types, final=True):
    m = _FUNC_RE.search(block)
    if not m:
        return None
    name = m.group(1).strip()
    if not name:
        return None
    body = block[m.end():]
    cut = body.find("</function>")
    if cut >= 0:
        body = body[:cut]
    elif not final:
        params, pending = _scan_params(body, final=False)
        return name, params, pending
    return name, _parse_params(body, final=final), None

def build_arguments(name, params, types):
    d = {}
    fn_types = types.get(name, {})
    for k, raw in params:
        d[k] = coerce(raw, fn_types.get(k))
    return json.dumps(d, ensure_ascii=False)

def split_tool_calls(text, tools=None, make_id=new_call_id, json_only=False):
    types = schema_types(tools)
    calls, gaps, pos, truncated = [], [], 0, False
    for m in _BLOCK_RE.finditer(text):
        parsed = _parse_block(m.group(1), types)
        if parsed is None:
            truncated = True
            continue
        gaps.append(text[pos:m.start()])
        pos = m.end()
        name, params, _ = parsed
        calls.append({"id": make_id(), "type": "function",
                      "function": {"name": name,
                                   "arguments": build_arguments(name, params,
                                                                types)}})
    tail = text[pos:]
    if TC_OPEN in tail:
        truncated = True
    gaps.append(tail)
    content = "".join(gaps).strip()
    if json_only and not calls and TC_OPEN not in text:
        k = content.find("{")
        content = content[k:] if k >= 0 else ""
    return (content or None), calls, truncated

class ToolStream:

    def __init__(self, tools=None, make_id=new_call_id, json_only=False):
        self.json_only = json_only
        self.types = schema_types(tools)
        self.make_id = make_id
        self.sent = ""
        self.calls = []

    @staticmethod
    def _esc(s):
        return json.dumps(s, ensure_ascii=False)[1:-1]

    def _outside(self, text):
        gaps, pos = [], 0
        for m in _BLOCK_RE.finditer(text):
            gaps.append(text[pos:m.start()])
            pos = m.end()
        tail = text[pos:]
        k = tail.find(TC_OPEN)
        if k >= 0:
            gaps.append(tail[:k])
        else:
            hold = 0
            for n in range(len(TC_OPEN) - 1, 0, -1):
                if tail.endswith(TC_OPEN[:n]):
                    hold = n
                    break
            gaps.append(tail[:len(tail) - hold])
        out = "".join(gaps)
        if self.json_only and TC_OPEN not in text:
            k = out.find("{")
            out = out[k:] if k >= 0 else ""
        return out

    def _blocks(self, text):
        out, pos = [], 0
        for m in _BLOCK_RE.finditer(text):
            out.append((m.group(1), True))
            pos = m.end()
        tail = text[pos:]
        k = tail.find(TC_OPEN)
        if k >= 0:
            out.append((tail[k + len(TC_OPEN):], False))
        return out

    def push(self, full):
        out = self._outside(full).strip()
        if out.startswith(self.sent):
            delta = out[len(self.sent):]
        else:
            n = 0
            while n < len(self.sent) and n < len(out) and self.sent[n] == out[n]:
                n += 1
            delta = out[n:]
        self.sent = out

        events = []
        for i, (body, complete) in enumerate(self._blocks(full)):
            parsed = _parse_block(body, self.types, final=complete)
            if parsed is None:
                continue
            name, params, pending = parsed
            while len(self.calls) <= i:
                self.calls.append(None)
            st = self.calls[i]
            if st is None:
                cid = self.make_id()
                st = self.calls[i] = {"name": name, "n": 0, "closed": False,
                                      "open": None, "v": 0, "id": cid}
                events.append({"index": i, "id": cid,
                               "type": "function",
                               "function": {"name": name, "arguments": ""}})
            if st["closed"]:
                continue
            fn_types = self.types.get(name, {})

            def emit(frag):
                if frag:
                    events.append({"index": i,
                                   "function": {"arguments": frag}})
            for k, raw in params[st["n"]:]:
                if st["open"] is not None:
                    emit(self._esc(raw[st["v"]:]) + '"')
                    st["open"], st["v"] = None, 0
                else:
                    emit(("{" if st["n"] == 0 else ", ")
                         + json.dumps(k, ensure_ascii=False) + ": "
                         + json.dumps(coerce(raw, fn_types.get(k)),
                                      ensure_ascii=False))
                st["n"] += 1
            if pending is not None and fn_types.get(pending[0]) == "string":
                k, sofar = pending
                if st["open"] is None:
                    emit(("{" if st["n"] == 0 else ", ")
                         + json.dumps(k, ensure_ascii=False) + ': "')
                    st["open"], st["v"] = k, 0
                safe = _stream_safe(sofar)
                if len(safe) > st["v"]:
                    emit(self._esc(safe[st["v"]:]))
                    st["v"] = len(safe)
            if complete:
                events.append({"index": i,
                               "function": {"arguments":
                                            "{}" if st["n"] == 0 else "}"}})
                st["closed"] = True
        return delta, events

    def any_calls(self):
        return any(c and c["closed"] for c in self.calls)

def normalize_messages(messages):
    msgs = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        m = dict(m)
        if m.get("role") == "function":
            m["role"] = "tool"
        if m.get("role") == "developer":
            m["role"] = "system"
        tcs = m.get("tool_calls")
        if m.get("role") == "assistant" and isinstance(tcs, list):
            fixed = []
            for tc in tcs:
                tc = dict(tc) if isinstance(tc, dict) else {}
                fn = tc.get("function")
                fn = dict(fn) if isinstance(fn, dict) else tc
                args = fn.get("arguments")
                if isinstance(args, str):
                    s = args.strip()
                    try:
                        args = json.loads(s) if s else {}
                    except Exception as e:
                        raise ValueError(
                            f"tool_calls[].function.arguments for "
                            f"'{fn.get('name')}' is not valid JSON: {e}")
                    if not isinstance(args, dict):
                        raise ValueError(
                            f"tool_calls[].function.arguments for "
                            f"'{fn.get('name')}' must decode to an object")
                    fn["arguments"] = args
                elif args is None:
                    fn["arguments"] = {}
                tc["function"] = fn
                fixed.append(tc)
            m["tool_calls"] = fixed
        msgs.append(m)

    def system_text(c):
        if isinstance(c, list):
            texts = []
            for part in c:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError("a system message may carry text only; "
                                     "send images in a user turn")
                texts.append(part.get("text") or "")
            c = "".join(texts)
        return str(c) if c else ""

    lead = 0
    while lead < len(msgs) and msgs[lead].get("role") == "system":
        lead += 1
    if lead > 1:
        parts = [t for t in (system_text(m.get("content")) for m in msgs[:lead]) if t]
        msgs[:lead] = [{"role": "system", "content": "\n\n".join(parts)}]
        lead = 1

    def place(text):
        prev = out[-1] if out else None
        if prev is not None and prev.get("role") == "user":
            c = prev.get("content")
            if isinstance(c, list):
                sep = "\n\n" if c else ""
                prev["content"] = list(c) + [{"type": "text", "text": sep + text}]
            else:
                prev["content"] = (c + "\n\n" + text) if c else text
        else:
            out.append({"role": "user", "content": text})

    out, late = msgs[:lead], []
    for m in msgs[lead:]:
        if m.get("role") == "system":
            t = system_text(m.get("content"))
            if t:
                late.append(t)
            continue
        if late and m.get("role") != "tool":
            place("\n\n".join(late))
            late = []
        out.append(m)
    if late:
        place("\n\n".join(late))
    msgs = out

    merged = []
    for m in msgs:
        prev = merged[-1] if merged else None
        if (prev is not None and m.get("role") == "assistant"
                and prev.get("role") == "assistant"
                and isinstance(m.get("content") or "", str)
                and isinstance(prev.get("content") or "", str)):
            a, b = prev.get("content") or "", m.get("content") or ""
            prev["content"] = a + "\n\n" + b if a and b else (a or b)
            ra, rb = prev.get("reasoning_content"), m.get("reasoning_content")
            if isinstance(rb, str) and rb.strip():
                prev["reasoning_content"] = (ra + "\n\n" + rb if isinstance(ra, str)
                                             and ra.strip() else rb)
            if isinstance(m.get("tool_calls"), list) and m["tool_calls"]:
                prev["tool_calls"] = list(prev.get("tool_calls") or []) + m["tool_calls"]
            continue
        merged.append(m)
    msgs = merged

    i = 0
    while i < len(msgs):
        if msgs[i].get("role") != "tool":
            i += 1
            continue
        j = i
        while j < len(msgs) and msgs[j].get("role") == "tool":
            j += 1
        prev = msgs[i - 1] if i else None
        order = {}
        if prev and isinstance(prev.get("tool_calls"), list):
            for n, tc in enumerate(prev["tool_calls"]):
                cid = tc.get("id") if isinstance(tc, dict) else None
                if isinstance(cid, str):
                    order[cid] = n
        run = msgs[i:j]
        ids = [r.get("tool_call_id") for r in run]
        if order and all(x in order for x in ids) and len(set(ids)) == len(ids):
            msgs[i:j] = sorted(run, key=lambda r: order[r["tool_call_id"]])
        i = j
    return msgs
