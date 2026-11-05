#!/usr/bin/env python3
"""
Incremental, 100% free Android strings.xml translator (no API key / card).
Uses the keyless Google Translate endpoint via the `deep-translator` package.

    pip install deep-translator
    python translate_strings.py                # every language below
    python translate_strings.py es de          # only these Android suffixes
    python translate_strings.py --list         # language suffixes as JSON
    python translate_strings.py --name es      # prints "Spanish"

Guarantees
  * Never writes the English text as a "translation" because a request failed.
    A failed string is retried (several rounds, growing waits). If any string
    is STILL missing at the end, the script exits with code 1 so nothing
    incomplete gets pushed.
  * Placeholders (%1$s, %d, \\n, \\', <b>, <xliff:g>, &amp;, @string/, {x}) and
    glossary terms are protected. If one translation style damages them, other
    styles are tried, and finally a segment-by-segment fallback that cannot
    damage them.
  * Multi-word strings that come back identical to English are retried with
    another style and reported as "suspicious" if they stay identical.
  * INCREMENTAL: progress lives in .translate-state/<lang>.json. Only new or
    edited English strings are translated; deleted ones are removed; strings
    that already had a hand-written translation are never touched.

Optional translate_glossary.txt (repo root): one term per line to never translate.
Env knobs: WORKERS (default 4), MAX_MINUTES (default 300), ROUNDS (default 4).
"""
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Android folder suffix -> Google Translate language code
LANGUAGES = {
    "es": "es",    # Spanish
    "de": "de",    # German
    "ru": "ru",    # Russian
    "fil": "tl",   # Filipino (Android folder values-fil, Google code tl)
    "ar": "ar",    # Arabic
    "id": "id",    # Indonesian
}
LANG_NAMES = {"es": "Spanish", "de": "German", "ru": "Russian",
              "fil": "Filipino", "ar": "Arabic", "id": "Indonesian"}

SKIP_DIRS = {"build", ".git", "out", "node_modules", ".gradle"}
STATE_DIR = Path(".translate-state")
GLOSSARY_FILE = Path("translate_glossary.txt")
WORKERS = int(os.environ.get("WORKERS", "4"))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "300"))
ROUNDS = int(os.environ.get("ROUNDS", "4"))

BLOCK_RE = re.compile(r"<(string|string-array|plurals)\b([^>]*?)>(.*?)</\1>", re.DOTALL)
ITEM_RE = re.compile(r"(<item\b[^>]*>)(.*?)(</item>)", re.DOTALL)
NAME_RE = re.compile(r'\bname\s*=\s*"([^"]+)"')
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

_BASE_PROTECT = (
    r"<[^>]+>"                                         # tags incl. <xliff:g>, <b>
    r"|<!\[CDATA\[|\]\]>"
    r"|%(?:\d+\$)?[-#+ 0,(]*\d*(?:\.\d+)?[a-zA-Z%]"    # %s %1$s %d %.2f
    r"|\\[nrt'\"@?\\]|\\u[0-9a-fA-F]{4}"               # \n \' \" \uXXXX
    r"|&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;"            # entities
    r"|@(?:string|plurals|array)/\w+"
    r"|\{[^{}]*\}"                                     # {placeholders}
)
PROTECT_RE = re.compile(_BASE_PROTECT)

# Two different token styles; translators mangle one or the other sometimes.
TOKEN_STYLES = [
    (lambda i: f"XQ{i}Z", lambda i: rf"XQ\s*{i}\s*Z"),
    (lambda i: f"[[{i}]]", lambda i: rf"\[\s*\[\s*{i}\s*\]\s*\]"),
]


def load_glossary():
    global PROTECT_RE
    if not GLOSSARY_FILE.exists():
        return
    terms = [t.strip() for t in GLOSSARY_FILE.read_text(encoding="utf-8").splitlines()
             if t.strip() and not t.startswith("#")]
    if terms:
        terms.sort(key=len, reverse=True)
        alt = "|".join(re.escape(t) for t in terms)
        PROTECT_RE = re.compile(_BASE_PROTECT + "|" + alt)
        print(f"Glossary: {len(terms)} protected term(s)")


def make_translator(code):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="en", target=code).translate


# ---------------------------------------------------------------- protection
def protect(text, style):
    """Swap placeholders for tokens. A pad space is added next to a token only
    where the original had none; restore() removes exactly those pads."""
    mk = style[0]
    saved = []

    def sub(m):
        i = len(saved)
        before = text[m.start() - 1] if m.start() > 0 else " "
        after = text[m.end()] if m.end() < len(text) else " "
        pre = "" if before.isspace() else " "
        post = "" if after.isspace() else " "
        saved.append((m.group(0), bool(pre), bool(post)))
        return f"{pre}{mk(i)}{post}"

    return PROTECT_RE.sub(sub, text), saved


def restore(text, saved, style):
    rx = style[1]
    for i, (tok, pre, post) in enumerate(saved):
        pat = re.compile((" ?" if pre else "") + rx(i) + (" ?" if post else ""), re.IGNORECASE)
        if not pat.search(text):
            return None  # token lost -> this style is unsafe
        text = pat.sub(lambda _m, t=tok: t, text, count=1)
    return text


def word_count(t):
    return len(re.findall(r"[A-Za-z]{2,}", PROTECT_RE.sub(" ", t)))


def request(translate, text):
    err = None
    for attempt in range(6):
        try:
            out = translate(text)
            if out:
                return out
            err = "empty response"
        except Exception as e:  # rate limit / network hiccup
            err = e
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"request failed: {err}")


def segment_translate(core, translate):
    """Fallback: translate only the plain-text pieces between placeholders."""
    out = []
    for p in re.split("(" + PROTECT_RE.pattern + ")", core):
        if not p:
            continue
        if PROTECT_RE.fullmatch(p) or not re.search(r"[A-Za-z]{2,}", p):
            out.append(p)
            continue
        lead, trail = p[: len(p) - len(p.lstrip())], p[len(p.rstrip()):]
        out.append(lead + request(translate, p.strip()).strip() + trail)
    return "".join(out)


def translate_text(text, translate, memo):
    """Always returns a real translation, or raises. Never returns None."""
    if text in memo:
        return memo[text]
    if not text.strip() or word_count(text) == 0:
        return text  # only placeholders / numbers / glossary terms
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    core = text.strip()
    first_ok = None
    for style in TOKEN_STYLES:
        protected, saved = protect(core, style)
        restored = restore(request(translate, protected), saved, style)
        if restored is None:
            continue
        restored = restored.strip()
        if restored.lower() != core.lower() or word_count(core) < 2:
            first_ok = restored
            break
        first_ok = first_ok or restored      # identical to English: try other style
    if first_ok is None:
        first_ok = segment_translate(core, translate).strip()
    result = lead + first_ok + trail
    memo[text] = result
    return result


def translate_block(kind, attrs, inner, translate, memo):
    if kind == "string":
        new_inner = translate_text(inner, translate, memo)
    else:
        new_inner = ITEM_RE.sub(
            lambda m: m.group(1) + translate_text(m.group(2), translate, memo) + m.group(3),
            inner)
    return f"<{kind}{attrs}>{new_inner}</{kind}>"


# ------------------------------------------------------------------- helpers
def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def strip_comments(t):
    return COMMENT_RE.sub(lambda m: " " * len(m.group(0)), t)


def block_re_for(name):
    return re.compile(
        r'<(string|string-array|plurals)\b[^>]*\bname\s*=\s*"%s"[^>]*>.*?</\1>'
        % re.escape(name), re.DOTALL)


def blocks_of(text):
    """Yield (kind, attrs, inner, name) for every translatable block."""
    for m in BLOCK_RE.finditer(strip_comments(text)):
        kind, attrs, inner = m.groups()
        nm = NAME_RE.search(attrs)
        if nm and not re.search(r'translatable\s*=\s*"false"', attrs):
            yield kind, attrs, inner, nm.group(1)


def names_in(text):
    out = set()
    for m in BLOCK_RE.finditer(strip_comments(text)):
        nm = NAME_RE.search(m.group(2))
        if nm:
            out.add(nm.group(1))
    return out


def load_state(suffix):
    p = STATE_DIR / f"{suffix}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(suffix, state):
    STATE_DIR.mkdir(exist_ok=True)
    (STATE_DIR / f"{suffix}.json").write_text(
        json.dumps(state, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")


# ------------------------------------------------------------------- per file
def process(src, suffix, code, factory, state, deadline):
    target = src.parent.parent / f"values-{suffix}" / "strings.xml"
    raw_src = src.read_text(encoding="utf-8")
    tgt_text = target.read_text(encoding="utf-8") if target.exists() else None
    have = names_in(tgt_text) if tgt_text else set()
    st = state.setdefault(src.as_posix(), {})

    work, src_names = [], set()
    for kind, attrs, inner, name in blocks_of(raw_src):
        h = sha(kind + inner)
        src_names.add(name)
        if name in have:
            if name not in st or st[name] == h:
                continue            # hand-written, or already up to date
            mode = "update"         # English text changed since we translated it
        else:
            mode = "new"
        attrs = re.sub(r'\s+translatable\s*=\s*"true"', "", attrs)
        work.append((name, kind, attrs, inner, h, mode))

    memo = {}

    def run(item):
        name, kind, attrs, inner, h, mode = item
        if time.time() > deadline:
            return item, None, "time budget reached"
        try:
            return item, translate_block(kind, attrs, inner, factory(code), memo), ""
        except Exception as e:
            return item, None, str(e)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(run, work))

    text, new_blocks = tgt_text, []
    ok, bad, suspicious = 0, 0, []
    for (name, kind, attrs, inner, h, mode), block, why in results:
        if block is None:
            bad += 1
            print(f"  ! {name}: {why}")
            continue
        ok += 1
        st[name] = h
        if block == f"<{kind}{attrs}>{inner}</{kind}>" and word_count(inner) >= 2:
            suspicious.append(name)
        if mode == "update":
            text = block_re_for(name).sub(lambda _m, b=block: b, text, count=1)
        else:
            new_blocks.append("    " + block)

    pruned = 0
    if text is not None:                       # drop removed English strings
        for name in [n for n in st if n not in src_names]:
            if name in have:
                full = re.compile(r"[ \t]*" + block_re_for(name).pattern + r"[ \t]*\n?", re.DOTALL)
                text = full.sub("", text, count=1)
                pruned += 1
            del st[name]

    if new_blocks:
        body = "\n".join(new_blocks) + "\n"
        if text is None:
            rm = re.search(r"<resources[^>]*>", raw_src)
            text = ('<?xml version="1.0" encoding="utf-8"?>\n'
                    + (rm.group(0) if rm else "<resources>") + "\n" + body
                    + "</resources>\n")
        else:
            head, _, tail = text.rpartition("</resources>")
            text = head + body + "</resources>" + tail

    if text is not None and text != tgt_text:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return {"ok": ok, "bad": bad, "pruned": pruned, "suspicious": suspicious}


def missing_strings(src, suffix):
    """Names of translatable English strings that have no entry in the target."""
    target = src.parent.parent / f"values-{suffix}" / "strings.xml"
    have = names_in(target.read_text(encoding="utf-8")) if target.exists() else set()
    return [n for _, _, _, n in blocks_of(src.read_text(encoding="utf-8")) if n not in have]


def summary(line):
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main(argv, factory=make_translator):
    if "--list" in argv:
        print(json.dumps(list(LANGUAGES)))
        return 0
    if "--name" in argv:
        print(LANG_NAMES.get(argv[argv.index("--name") + 1], "?"))
        return 0
    wanted = argv or [s for s in os.environ.get("LANGS", "").replace(",", " ").split() if s]
    unknown = [w for w in wanted if w not in LANGUAGES]
    if unknown:
        print("Unknown language suffix(es):", unknown, "- choose from", list(LANGUAGES))
        return 1
    langs = {k: v for k, v in LANGUAGES.items() if not wanted or k in wanted}
    sources = sorted(
        p for p in Path(".").rglob("res/values/strings.xml")
        if not (set(p.parts) & SKIP_DIRS)
    )
    if not sources:
        print("No res/values/strings.xml found under", Path.cwd())
        return 1
    load_glossary()
    deadline = time.time() + MAX_MINUTES * 60
    rc = 0
    for suffix, code in langs.items():
        state = load_state(suffix)
        total_ok = total_pruned = 0
        suspicious = []
        for src in sources:
            for rnd in range(1, ROUNDS + 1):
                r = process(src, suffix, code, factory, state, deadline)
                total_ok += r["ok"]
                total_pruned += r["pruned"]
                suspicious += [f"{src.parent.parent.parent.name}:{n}" for n in r["suspicious"]]
                save_state(suffix, state)       # save after every pass -> resumable
                if r["bad"] == 0:
                    break
                print(f"  round {rnd}/{ROUNDS}: {r['bad']} string(s) failed, retrying...")
                time.sleep(min(20 * rnd, 60))
            miss = missing_strings(src, suffix)
            print(f"[{suffix}] {src}: +{total_ok} translated, {total_pruned} removed, "
                  f"{len(miss)} missing")
            if miss:
                rc = 1
                print(f"  INCOMPLETE: {len(miss)} string(s) not translated: {miss[:20]}")
        summary(f"**{LANG_NAMES.get(suffix, suffix)} ({suffix})**: {total_ok} translated"
                + (f", ⚠️ {len(suspicious)} unchanged-English (acronyms/names?)" if suspicious else ""))
        if suspicious:
            print(f"[{suffix}] unchanged-English multi-word strings (review): {suspicious[:30]}")
        # Translator silently returning English for many strings = service problem.
        if total_ok >= 5 and len(suspicious) / total_ok > 0.2:
            rc = 1
            print(f"[{suffix}] FAILED: {len(suspicious)} of {total_ok} strings came back "
                  f"identical to English. The translation service is probably not working "
                  f"right now - nothing should be pushed. Re-run later.")
            summary(f"❌ {LANG_NAMES.get(suffix, suffix)}: translator returned English for "
                    f"{len(suspicious)}/{total_ok} strings - failed on purpose, re-run later")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
