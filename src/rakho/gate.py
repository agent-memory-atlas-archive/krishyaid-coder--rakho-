from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from rakho.models import Decision


class PolicyGate(ABC):
    """Inspect content before it is stored. Sluice (or anything else) can implement this."""

    @abstractmethod
    def inspect(self, content: str, context: dict[str, Any] | None = None) -> Decision:
        raise NotImplementedError


class AllowAllGate(PolicyGate):
    """Tests only. The default gate is DefaultPolicyGate."""

    def inspect(self, content: str, context: dict[str, Any] | None = None) -> Decision:
        return Decision(action="store", content=content, reason="allow_all")


_INJECTION = re.compile(
    r"(ignore (all )?(the )?(previous|prior|above|earlier) (instructions?|directions?|prompts?|rules)"
    r"|forget (the |all )?safety"
    r"|disregard (all |your |the )?(previous|prior|earlier|above|rules|guidelines|safety|directions?)"
    r"|do not follow (any |the )?(previous|prior|earlier) (instructions?|rules|prompts?)"
    r"|you are now (dan|jailbreak|unrestricted|jailbroken)"
    r"|enter dan mode"
    r"|override (the )?(system|safety) prompt"
    r"|system prompt is"
    r"|ignoriere (alle )?(vorherigen?|bisherigen?|obigen?) anweisungen"
    r"|vergiss (die )?(sicherheitsregeln|vorherigen? anweisungen)"
    r"|पिछले निर्देशों को (अनदेखा|नज़रअंदाज़|नजरअंदाज)"
    r"|पिछली (हिदायतों|निर्देशों) को (भूल|अनदेखा))",
    re.IGNORECASE,
)

# Replacement token is never a real secret.
_MASK = "[REDACTED:{label}]"

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN ([A-Z ]+)?PRIVATE KEY-----[\s\S]+?-----END ([A-Z ]+)?PRIVATE KEY-----")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._\-+=/]{12,}\b", re.IGNORECASE)),
    (
        "password",
        re.compile(
            r"\b(password|passwd|pwd|passphrase|secret|api[_-]?key|auth[_-]?token"
            r"|passwort|kennwort|geheimnis|api[-_]?schl[uü]ssel)\b"
            r"\s*([:=]|\bist\b|\bis\b)\s*\S+",
            re.IGNORECASE,
        ),
    ),
    (
        "password",
        re.compile(r"(पासवर्ड|कूटशब्द|गुप्त(?:शब्द|कोड)?)\s*(?:[:=]|है)?\s+\S+"),
    ),
]

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("iban", re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("aadhaar", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    (
        "steuer_id",
        re.compile(
            r"(?i)(?:steuer(?:liche)?(?:\s|-)?(?:id(?:entifikationsnummer)?|"
            r"identifikationsnummer)|idnr|steuernummer)"
            r"\s*[:=]?\s*([1-9]\d{10}|\d{2}\s\d{3}\s\d{3}\s\d{3})"
        ),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")),
]


def _luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _useful_remainder(text: str) -> str:
    cleaned = re.sub(r"\[REDACTED:[a-z_]+\]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :;,-.")
    return cleaned.strip()


class DefaultPolicyGate(PolicyGate):
    """Fail-closed regex gate: secrets, obvious PII, injection-shaped content.

    Mixed notes are redacted and stored. Poisoned or secret-only notes are skipped.
    """

    def inspect(self, content: str, context: dict[str, Any] | None = None) -> Decision:
        if not content or not content.strip():
            return Decision(action="skip", content="", reason="empty", labels=["empty"])

        extra = _context_text(context)
        haystack = content if not extra else f"{content}\n{extra}"
        if _INJECTION.search(haystack):
            return Decision(
                action="skip",
                content="",
                reason="prompt_injection",
                labels=["prompt_injection"],
            )

        extra_hit = _scan_labels(extra) if extra else []
        if extra_hit:
            return Decision(
                action="skip",
                content="",
                reason="secret_or_pii",
                labels=extra_hit,
            )

        redacted = content
        labels: list[str] = []

        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(redacted):
                redacted = pattern.sub(_MASK.format(label=label), redacted)
                if label not in labels:
                    labels.append(label)

        for label, pattern in _PII_PATTERNS:
            def _sub(match: re.Match[str], label: str = label) -> str:
                raw = match.group(0)
                if label == "credit_card":
                    digits = re.sub(r"\D", "", raw)
                    if not _luhn_ok(digits):
                        return raw
                if label == "aadhaar":
                    digits = re.sub(r"\D", "", raw)
                    if len(digits) != 12 or _luhn_ok(digits):
                        return raw
                if label == "iban" and len(re.sub(r"\s", "", raw)) < 15:
                    return raw
                if label == "steuer_id":
                    digits = re.sub(r"\D", "", raw)
                    if len(digits) != 11 or digits.startswith("0"):
                        return raw
                return _MASK.format(label=label)

            new_text, n = pattern.subn(_sub, redacted)
            if n and new_text != redacted:
                redacted = new_text
                if label not in labels:
                    labels.append(label)

        if not labels:
            return Decision(action="store", content=content, reason="clean")

        remainder = _useful_remainder(redacted)
        if len(remainder) < 8:
            return Decision(
                action="skip",
                content="",
                reason="secret_or_pii",
                labels=labels,
            )

        return Decision(
            action="redact",
            content=redacted,
            reason="redacted:" + ",".join(labels),
            labels=labels,
        )


def _context_text(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    parts: list[str] = []
    for key in ("source", "origin_tool"):
        value = context.get(key)
        if value:
            parts.append(str(value))
    meta = context.get("metadata")
    if meta:
        parts.append(json.dumps(meta, ensure_ascii=False) if not isinstance(meta, str) else meta)
    return " ".join(parts)


def _scan_labels(text: str) -> list[str]:
    if not text.strip():
        return []
    redacted = text
    labels: list[str] = []
    for label, pattern in _SECRET_PATTERNS + _PII_PATTERNS:
        def _sub(match: re.Match[str], label: str = label) -> str:
            raw = match.group(0)
            if label == "credit_card":
                digits = re.sub(r"\D", "", raw)
                if not _luhn_ok(digits):
                    return raw
            if label == "aadhaar":
                digits = re.sub(r"\D", "", raw)
                if len(digits) != 12 or _luhn_ok(digits):
                    return raw
            if label == "iban" and len(re.sub(r"\s", "", raw)) < 15:
                return raw
            if label == "steuer_id":
                digits = re.sub(r"\D", "", raw)
                if len(digits) != 11 or digits.startswith("0"):
                    return raw
            return _MASK.format(label=label)

        new_text, n = pattern.subn(_sub, redacted)
        if n and new_text != redacted:
            redacted = new_text
            if label not in labels:
                labels.append(label)
    return labels


def is_injection(text: str) -> bool:
    return bool(text and _INJECTION.search(text))


def mask_preview(content: str, decision: Decision, limit: int = 80) -> str:
    """What audit is allowed to show for a refusal. Never the raw secret."""
    if decision.content:
        text = decision.content
    else:
        labels = ",".join(decision.labels) or decision.reason or "blocked"
        text = f"[refused:{labels}]"
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text
