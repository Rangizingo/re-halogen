#!/usr/bin/env python3
# halogen-flash-server: source shipped as-is, comments stripped. The README is the documentation.
import argparse
import base64
import collections
import hashlib
import hmac
import io
import json
import os
import queue
import re
import socket
import sys
import threading
import time

FIELDS = ("prompt", "reply", "reasoning", "args", "results")
ROTATE_BYTES = 64 << 20
QUEUE_MAX = 2048
HEX16 = re.compile(r"[0-9a-f]{32}")
CALL_ID = re.compile(r"call_[0-9a-f]{24}")

_CLIENTS = (("claude-code", re.compile(r"claude-(cli|code)|claude_code", re.I)),
            ("codex", re.compile(r"codex", re.I)),
            ("opencode", re.compile(r"opencode", re.I)),
            ("aider", re.compile(r"aider", re.I)),
            ("cline", re.compile(r"cline|roo", re.I)),
            ("kilo", re.compile(r"kilo", re.I)),
            ("continue", re.compile(r"continue", re.I)),
            ("hermes", re.compile(r"hermes", re.I)),
            ("anthropic-sdk", re.compile(r"anthropic", re.I)),
            ("openai-sdk", re.compile(r"openai", re.I)),
            ("curl", re.compile(r"^curl", re.I)),
            ("python-http", re.compile(r"python|httpx|aiohttp|requests", re.I)),
            ("node-http", re.compile(r"node|undici|axios", re.I)))

def client_family(ua):
    ua = ua or ""
    for name, rx in _CLIENTS:
        if rx.search(ua):
            return name
    return "other" if ua else "none"

def parse_text_fields(spec):
    s = (spec if spec is not None else "all").strip().lower()
    if s in ("", "all"):
        return frozenset(FIELDS)
    if s == "none":
        return frozenset()
    out = set()
    for p in s.split(","):
        p = p.strip()
        if p in FIELDS:
            out.add(p)
        elif p:
            raise ValueError("HALOGEN_RECORD_TEXT: unknown field %r (want all, none or a list of %s)"
                             % (p, ", ".join(FIELDS)))
    return frozenset(out)

def _canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _day(t):
    return time.strftime("%Y-%m-%d", time.gmtime(t))

class Rec:
    __slots__ = ("rc", "d", "done")

    def __init__(self, rc, d):
        self.rc, self.d, self.done = rc, d, False

    def admitted(self, cid, wait_ms, sig):
        self.d["id"] = cid
        self.d["wait_ms"] = int(wait_ms)
        self.d["req"] = sig

    def finish(self, finish, done=None, eng=None, reasoning="", body="",
               call_ids=(), tools=None, json_only=False):
        if self.done:
            return
        self.done = True
        try:
            d = self.d
            d["finish"] = finish
            d["n_gen"] = int((done or {}).get("n_gen", 0) or 0)
            d["n_reasoning"] = int((done or {}).get("n_reasoning", 0) or 0)
            d["eng"] = eng
            d["reasoning"], d["body"] = reasoning or "", body or ""
            d["call_ids"], d["tools"], d["json_only"] = list(call_ids), tools, json_only
            self.rc.submit(d)
        except Exception as e:
            self.rc.note("finish: %s" % e)

def from_env(log=print):
    d = os.environ.get("HALOGEN_RECORD_DIR", "").strip()
    if not d:
        return None
    try:
        rc = Recorder(d, gib=float(os.environ.get("HALOGEN_RECORD_GIB", "2") or 2),
                      days=int(os.environ.get("HALOGEN_RECORD_DAYS", "90") or 90),
                      text=os.environ.get("HALOGEN_RECORD_TEXT"), log=log)
    except (ValueError, OSError, RuntimeError) as e:
        raise SystemExit("serve_api: HALOGEN_RECORD_DIR is set but request records "
                         "cannot start: %s" % e)
    print("serve_api: request records ON in %s (text fields: %s; cap %s GiB, %s days)"
          % (rc.root, ",".join(sorted(rc.text)) or "none",
             os.environ.get("HALOGEN_RECORD_GIB", "2") or "2",
             os.environ.get("HALOGEN_RECORD_DAYS", "90") or "90"), flush=True)
    return rc

class Recorder:
    def __init__(self, root, gib=2.0, days=90, text=None, log=print):
        self.root = os.path.abspath(root)
        self.cap = int(float(gib) * (1 << 30))
        self.days = int(days)
        self.text = parse_text_fields(text)
        self.log = log
        self.q = queue.Queue(QUEUE_MAX)
        self.dropped = 0
        self.dropped_cap = 0
        self.written = 0
        self.bytes = 0
        self._said = set()
        self.store = None
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        self.store = _Store(self)
        self.thread = threading.Thread(target=self._run, name="record-writer", daemon=True)
        self.thread.start()

    def note(self, msg):
        if msg not in self._said and len(self._said) < 20:
            self._said.add(msg)
            try:
                self.log("serve_api: request records: %s" % msg, flush=True)
            except TypeError:
                self.log("serve_api: request records: %s" % msg)

    def begin(self, wire, http_request, sess, src, n_msgs, model, stream,
              effort=None, tool_choice=None):
        ua = ""
        try:
            ua = http_request.headers.get("user-agent", "") if http_request is not None else ""
        except Exception:
            pass
        d = {"t": time.time(), "wire": wire, "stream": bool(stream), "model": model,
             "client": client_family(ua), "sess": sess, "src": src,
             "n_msgs": int(n_msgs), "effort": effort, "tool_choice": tool_choice}
        return Rec(self, d)

    def submit_npu(self, job):
        job["npu"] = True
        self.submit(job)

    def submit(self, job):
        try:
            self.q.put_nowait(job)
        except queue.Full:
            self.dropped += 1
            self.note("the writer queue is full; records are being dropped (counted in /health)")

    def health(self):
        return {"dir": True, "written": self.written, "dropped": self.dropped,
                "dropped_cap": self.dropped_cap, "bytes": self.store.total,
                "text": sorted(self.text)}

    def flush(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.q.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    def _run(self):
        while True:
            job = self.q.get()
            try:
                self.store.write(job)
            except Exception as e:
                self.dropped += 1
                self.note("write failed: %s" % e)
            finally:
                self.q.task_done()

class _Store:
    def __init__(self, rc):
        self.rc = rc
        self.root = rc.root
        self.salt = self._salt()
        self.turns = os.path.join(self.root, "turns")
        self.units = os.path.join(self.root, "units")
        os.makedirs(self.turns, mode=0o700, exist_ok=True)
        os.makedirs(self.units, mode=0o700, exist_ok=True)
        self.host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())[:40]
        self.fh = None
        self.fday = None
        self.fbytes = 0
        self.fname = None
        self.seen = set()
        self.ours = collections.OrderedDict()
        self.total = self._scan()
        self._prune_days()
        self._warm()

    def _salt(self):
        p = os.path.join(self.root, ".salt")
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(os.urandom(32))
        except FileExistsError:
            pass
        with open(p, "rb") as f:
            s = f.read()
        if len(s) != 32:
            raise RuntimeError("the store's salt file is damaged (%s)" % p)
        return s

    def _scan(self):
        n = 0
        for dp, _, fs in os.walk(self.root):
            for f in fs:
                if f == ".salt":
                    continue
                try:
                    n += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        return n

    def _warm(self):
        k = 0
        for dp, _, fs in os.walk(self.units):
            for f in fs:
                if f.endswith(".json"):
                    self.seen.add(f[:-5])
                    k += 1
            if k > 500000:
                break
        days = sorted(os.listdir(self.turns))[-2:]
        for dd in days:
            p = os.path.join(self.turns, dd)
            for f in sorted(os.listdir(p)):
                try:
                    with open(os.path.join(p, f), "r", encoding="utf-8", errors="replace") as fh:
                        for m in CALL_ID.finditer(fh.read()):
                            self._remember(m.group(0))
                except OSError:
                    pass

    def _remember(self, cid):
        self.ours[cid] = 1
        if len(self.ours) > 100000:
            self.ours.popitem(last=False)

    def hmac(self, tag, data, n):
        return hmac.new(self.salt, tag + b"\0" + data, hashlib.sha256).hexdigest()[:n]

    def sess_hash(self, sess):
        for src, val in sess or ():
            if val is None or val == "":
                continue
            v = val if isinstance(val, str) else _canon(val)
            return {"src": src, "h": self.hmac(b"s:" + src.encode(), v.encode("utf-8", "replace"), 16)}
        return None

    def _image(self, url):
        w = h = None
        try:
            if isinstance(url, str) and url.startswith("data:") and "," in url:
                raw = base64.b64decode(url.split(",", 1)[1], validate=False)
                from PIL import Image
                w, h = Image.open(io.BytesIO(raw)).size
        except Exception:
            pass
        u = url if isinstance(url, str) else _canon(url)
        out = {"type": "image", "h": self.hmac(b"i", u.encode("utf-8", "replace"), 16)}
        if w:
            out["w"], out["h_px"] = w, h
        return out

    def _content(self, c):
        if c is None:
            return ""
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"):
                    iu = p.get("image_url", p.get("source", ""))
                    if isinstance(iu, dict):
                        iu = iu.get("url", _canon(iu))
                    parts.append(self._image(iu))
                elif isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                else:
                    parts.append(_canon(p))
            if all(isinstance(x, str) for x in parts):
                return "".join(parts)
            return parts
        return _canon(c)

    @staticmethod
    def _args(a):
        if isinstance(a, str):
            try:
                return _canon(json.loads(a))
            except Exception:
                return a
        return _canon(a)

    def canon_msg(self, m):
        if not isinstance(m, dict):
            return {"role": "?", "content": _canon(m)}
        o = {"role": str(m.get("role", "")), "content": self._content(m.get("content"))}
        tcs = []
        for tc in m.get("tool_calls") or ():
            fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
            tcs.append({"id": (tc.get("id") if isinstance(tc, dict) else "") or "",
                        "name": fn.get("name", ""), "arguments": self._args(fn.get("arguments", ""))})
        if tcs:
            o["tool_calls"] = tcs
        if m.get("tool_call_id"):
            o["tool_call_id"] = m["tool_call_id"]
        return o

    def redact(self, o):
        t = self.rc.text
        r = dict(o)
        role = o["role"]
        field = ("results" if role == "tool" else "reply" if role == "assistant" else "prompt")

        def red(c):
            if field in t:
                return c
            if isinstance(c, str):
                return {"omitted": len(c)}
            return [x if isinstance(x, dict) else {"omitted": len(x)} for x in c]
        r["content"] = red(o["content"])
        if "tool_calls" in o and "args" not in t:
            r["tool_calls"] = [dict(tc, arguments={"omitted": len(tc["arguments"])}) for tc in o["tool_calls"]]
        return r

    def put_unit(self, o):
        h = self.hmac(b"u", _canon(o).encode("utf-8", "replace"), 32)
        if h in self.seen:
            return h, 0
        body = _canon(dict(self.redact(o), h=h)).encode("utf-8", "replace")
        d = os.path.join(self.units, h[:2])
        os.makedirs(d, mode=0o700, exist_ok=True)
        p = os.path.join(d, h + ".json")
        if os.path.exists(p):
            self.seen.add(h)
            return h, 0
        return h, (p, body)

    def write_npu(self, j):
        rec = {"v": 1, "t": round(j["t"], 3), "id": j["id"], "wire": "npu", "route": j["route"],
               "kind": j["kind"], "model": j.get("model"), "rev": j.get("rev"),
               "sess": self.sess_hash(j.get("sess")), "client": j.get("client", "none"),
               "status": j["status"], "latency_ms": j["latency_ms"], "req": j.get("req") or {},
               "dec": j.get("dec") or []}
        new = []
        if j.get("units_in") is not None:
            units = []
            for m in j["units_in"]:
                h, nu = self.put_unit(self.canon_msg(m))
                units.append(h)
                if nu:
                    new.append((h, nu))
            rec["units"] = units
            if j.get("out") is not None:
                rh, nu = self.put_unit({"role": "assistant", "content": j["out"]})
                rec["reply"] = rh
                if nu:
                    new.append((rh, nu))
        self._commit(rec, new, j["t"])

    def write(self, j):
        if j.get("npu"):
            return self.write_npu(j)
        from tool_parse import split_tool_calls
        t = self.rc.text
        rec = {"v": 1, "t": round(j["t"], 3), "id": j.get("id"), "wire": j["wire"],
               "stream": j["stream"], "model": j["model"],
               "sess": self.sess_hash(j["sess"]), "client": j["client"]}
        req = dict(j.get("req") or {})
        req["n_msgs"] = j["n_msgs"]
        req["effort"] = j["effort"]
        req["tool_choice"] = j["tool_choice"]
        rec["req"] = req
        raw_msgs, ins = self._read_src(j["src"])
        rec["in"] = [{"call_id": (i["call_id"] if i["call_id"] in self.ours else None),
                      "is_error": i["is_error"]} for i in ins]
        calls, content = [], j["body"]
        if j["tools"]:
            ids = iter(j["call_ids"])
            content, parsed, _ = split_tool_calls(j["body"], j["tools"],
                                                  make_id=lambda: next(ids, ""),
                                                  json_only=j["json_only"])
            calls = parsed[:len(j["call_ids"])] if j["call_ids"] else []
        rec["out"] = {"finish": j["finish"], "n_gen": j["n_gen"], "n_reasoning": j["n_reasoning"],
                      "calls": [{"id": c["id"], "name": c["function"]["name"]} for c in calls]}
        rec["eng"] = j["eng"]
        rec["wait_ms"] = j.get("wait_ms")
        new = []
        if t and raw_msgs is not None:
            units = []
            for m in raw_msgs:
                h, nu = self.put_unit(self.canon_msg(m))
                units.append(h)
                if nu:
                    new.append((h, nu))
            reply = {"role": "assistant", "content": content or "",
                     "tool_calls": [{"id": c["id"], "name": c["function"]["name"],
                                     "arguments": self._args(c["function"]["arguments"])} for c in calls]}
            if not reply["tool_calls"]:
                del reply["tool_calls"]
            rh, nu = self.put_unit(reply)
            if nu:
                new.append((rh, nu))
            rec["units"], rec["reply"] = units, rh
            if j["reasoning"]:
                zh, nu = self.put_unit({"role": "reasoning", "content": j["reasoning"]})
                rec["reasoning"] = zh
                if nu:
                    new.append((zh, nu))
        for c in calls:
            self._remember(c["id"])
        self._commit(rec, new, j["t"])

    def _commit(self, rec, new, t):
        line = (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8", "replace")
        need = len(line) + sum(len(b) for _, (_, b) in new)
        if self.total + need > self.rc.cap:
            self.rc.dropped_cap += 1
            self.rc.note("the store is at its size cap (HALOGEN_RECORD_GIB); new records are being dropped")
            return
        for h, (p, b) in new:
            tmp = p + ".tmp"
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
                f.write(b)
            os.replace(tmp, p)
            self.seen.add(h)
            self.total += len(b)
        self._append(t, line)
        self.rc.written += 1

    def _read_src(self, src):
        try:
            kind = src[0]
            if kind == "chat":
                msgs = list(src[1])
                ins = []
                for m in reversed(msgs):
                    if isinstance(m, dict) and m.get("role") == "tool":
                        ins.append({"call_id": m.get("tool_call_id"), "is_error": None})
                    else:
                        break
                return msgs, ins[::-1]
            if kind == "messages":
                system, turns, messages_to_openai = src[1], src[2], src[3]
                ins = []
                last = turns[-1] if turns else None
                if isinstance(last, dict) and last.get("role") == "user" and isinstance(last.get("content"), list):
                    for b in last["content"]:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            ins.append({"call_id": b.get("tool_use_id"), "is_error": bool(b.get("is_error"))})
                return messages_to_openai(system, turns), ins
            if kind == "responses":
                responses_messages = src[3]
                items = src[2] if isinstance(src[2], list) else []
                ins = []
                for it in reversed(items):
                    if isinstance(it, dict) and it.get("type") == "function_call_output":
                        ins.append({"call_id": it.get("call_id"), "is_error": None})
                    else:
                        break
                return responses_messages(src[1], src[2]), ins[::-1]
        except Exception as e:
            self.rc.note("could not read a request's messages: %s" % e)
        return None, []

    def _append(self, t, line):
        day = _day(t)
        if self.fh is not None and (day != self.fday or self.fbytes + len(line) > ROTATE_BYTES):
            self.fh.close()
            self.fh = None
            if day != self.fday:
                self._prune_days()
        if self.fh is None:
            d = os.path.join(self.turns, day)
            os.makedirs(d, mode=0o700, exist_ok=True)
            self.fname = os.path.join(d, "%s-%d-%d.jsonl" % (self.host, os.getpid(), int(time.time() * 1000)))
            self.fh = os.fdopen(os.open(self.fname, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "ab")
            self.fday, self.fbytes = day, 0
        self.fh.write(line)
        self.fh.flush()
        self.fbytes += len(line)
        self.total += len(line)

    def _prune_days(self):
        if self.rc.days <= 0:
            return
        n = prune_days(self.root, self.rc.days, keep=self.fname)
        if n:
            self.total = self._scan()

def iter_records(root):
    td = os.path.join(root, "turns")
    if not os.path.isdir(td):
        return
    for dd in sorted(os.listdir(td)):
        p = os.path.join(td, dd)
        for f in sorted(os.listdir(p)):
            try:
                with open(os.path.join(p, f), "r", encoding="utf-8", errors="replace") as fh:
                    for ln in fh:
                        try:
                            r = json.loads(ln)
                        except Exception:
                            continue
                        if isinstance(r, dict):
                            yield r
            except OSError:
                pass

def referenced_units(root, skip=None):
    td = os.path.join(root, "turns")
    refs = set()
    if not os.path.isdir(td):
        return refs
    for dd in os.listdir(td):
        for f in os.listdir(os.path.join(td, dd)):
            try:
                with open(os.path.join(td, dd, f), "r", encoding="utf-8", errors="replace") as fh:
                    refs.update(HEX16.findall(fh.read()))
            except OSError:
                pass
    return refs

def gc_units(root):
    refs = referenced_units(root)
    n = b = 0
    ud = os.path.join(root, "units")
    for dp, _, fs in os.walk(ud):
        for f in fs:
            if f.endswith(".json") and f[:-5] not in refs:
                p = os.path.join(dp, f)
                try:
                    b += os.path.getsize(p)
                    os.unlink(p)
                    n += 1
                except OSError:
                    pass
    return n, b

def prune_days(root, days, keep=None):
    td = os.path.join(root, "turns")
    if not os.path.isdir(td):
        return 0
    cutoff = time.time() - days * 86400
    n = 0
    for dd in sorted(os.listdir(td)):
        p = os.path.join(td, dd)
        for f in os.listdir(p):
            fp = os.path.join(p, f)
            if keep and os.path.abspath(fp) == os.path.abspath(keep):
                continue
            try:
                if os.path.getmtime(fp) < cutoff:
                    os.unlink(fp)
                    n += 1
            except OSError:
                pass
        try:
            os.rmdir(p)
        except OSError:
            pass
    if n:
        gc_units(root)
    return n

def _store_size(root):
    n = 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f != ".salt":
                try:
                    n += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
    return n

def cmd_stats(root):
    rows = list(iter_records(root))
    c = collections.Counter
    by = lambda k: dict(c(k(r) for r in rows))
    ours_seen = set()
    for r in rows:
        for x in r.get("out", {}).get("calls", ()):
            ours_seen.add(x["id"])
    ins = [i for r in rows for i in r.get("in", ())]
    answered = [i for i in ins if i.get("call_id")]
    err = [i for i in answered if i.get("is_error")]
    cached = tot = 0
    for r in rows:
        e = r.get("eng") or {}
        cached += int(e.get("cache_n", 0) or 0)
        tot += int(e.get("cache_n", 0) or 0) + int(e.get("prompt_n", 0) or 0)
    ts = [r.get("t", 0) for r in rows]
    print(json.dumps({
        "records": len(rows), "store_bytes": _store_size(root),
        "oldest": time.strftime("%Y-%m-%d %H:%M", time.gmtime(min(ts))) if ts else None,
        "newest": time.strftime("%Y-%m-%d %H:%M", time.gmtime(max(ts))) if ts else None,
        "by_wire": by(lambda r: r.get("wire")), "by_client": by(lambda r: r.get("client")),
        "sessions": len({(r.get("sess") or {}).get("h") for r in rows if r.get("sess")}),
        "unattributed": sum(1 for r in rows if not r.get("sess")),
        "finish": by(lambda r: (r.get("out") or {}).get("finish")),
        "think": by(lambda r: (r.get("req") or {}).get("think")),
        "effort": by(lambda r: (r.get("req") or {}).get("effort")),
        "cache_hit_share": round(cached / tot, 4) if tot else None,
        "our_calls_answered": len(answered), "is_error": len(err),
        "is_error_rate": round(len(err) / len(answered), 4) if answered else None,
        "units": sum(len(fs) for _, _, fs in os.walk(os.path.join(root, "units"))),
    }, indent=1))

def _rewrite_without(root, drop):
    td = os.path.join(root, "turns")
    n = skipped = 0
    for dd in os.listdir(td):
        for f in os.listdir(os.path.join(td, dd)):
            fp = os.path.join(td, dd, f)
            if time.time() - os.path.getmtime(fp) < 120:
                skipped += 1
                continue
            keep, hit = [], 0
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                    except Exception:
                        keep.append(ln)
                        continue
                    if drop(r):
                        hit += 1
                    else:
                        keep.append(ln)
            if hit:
                tmp = fp + ".tmp"
                with open(tmp, "w", encoding="utf-8") as o:
                    o.writelines(keep)
                os.replace(tmp, fp)
                n += hit
    return n, skipped

def cmd_prune(a):
    root = a.dir
    if a.days is not None:
        print("removed %d file(s) older than %d day(s)" % (prune_days(root, a.days), a.days))
    if a.gib is not None:
        cap = int(a.gib * (1 << 30))
        files = []
        td = os.path.join(root, "turns")
        for dd in os.listdir(td) if os.path.isdir(td) else ():
            for f in os.listdir(os.path.join(td, dd)):
                fp = os.path.join(td, dd, f)
                files.append((os.path.getmtime(fp), fp))
        files.sort()
        while _store_size(root) > cap and files:
            os.unlink(files.pop(0)[1])
            gc_units(root)
        print("store is now %d bytes" % _store_size(root))
    if a.all:
        import shutil
        for sub in ("turns", "units") + (("pinned",) if a.pinned else ()):
            shutil.rmtree(os.path.join(root, sub), ignore_errors=True)
        print("removed the working store" + (" and the pinned slices" if a.pinned else ""))
    if a.unit:
        p = os.path.join(root, "units", a.unit[:2], a.unit + ".json")
        if os.path.exists(p):
            os.unlink(p)
            print("removed unit %s (records keep the hash)" % a.unit)
        else:
            print("no such unit")
    if a.session:
        n, sk = _rewrite_without(root, lambda r: (r.get("sess") or {}).get("h") == a.session)
        g, b = gc_units(root)
        print("removed %d record(s) of session %s, %d unit(s) (%d bytes); skipped %d file(s) written in the last 2 minutes"
              % (n, a.session, g, b, sk))
    if a.days is None and a.gib is None and not (a.all or a.unit or a.session):
        print("nothing to do: give --days, --gib, --all, --unit or --session", file=sys.stderr)
        return 2
    g, b = gc_units(root)
    if g:
        print("collected %d unreferenced unit(s)" % g)
    return 0

def main(argv=None):
    ap = argparse.ArgumentParser(prog="request_records")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stats")
    s.add_argument("dir")
    p = sp.add_parser("prune")
    p.add_argument("dir")
    p.add_argument("--days", type=int)
    p.add_argument("--gib", type=float)
    p.add_argument("--all", action="store_true")
    p.add_argument("--pinned", action="store_true")
    p.add_argument("--unit")
    p.add_argument("--session")
    a = ap.parse_args(argv)
    if a.cmd == "stats":
        cmd_stats(a.dir)
        return 0
    return cmd_prune(a)

if __name__ == "__main__":
    sys.exit(main())
