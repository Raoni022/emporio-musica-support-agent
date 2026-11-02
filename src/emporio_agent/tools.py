"""Business tools called by the agent.

Each tool is a plain function that takes its data dependencies explicitly
and returns a JSON-serializable dict. Errors are returned as
{"error": ..., "retryable": bool}, never raised to the LLM as a stack trace.
"""

from __future__ import annotations

from emporio_agent.policies import SECTION_NOTES, TOPICS, PolicyManual


def get_policy(manual: PolicyManual, topic: str) -> dict:
    """Return one section of the store policy manual, with its version."""
    if topic not in TOPICS:
        return {
            "error": f"Unknown policy topic: {topic!r}.",
            "valid_topics": sorted(TOPICS),
            "retryable": False,
        }
    section = manual.by_topic(topic)
    return {
        "section": section.number,
        "title": section.title,
        "version": manual.version,
        "text": section.text,
        "notes": SECTION_NOTES.get(section.number, []),
    }
