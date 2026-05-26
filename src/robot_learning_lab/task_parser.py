import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RobotCommand:
    action: str
    object: str
    target: Optional[str] = None


ENGLISH_PICK_PLACE = re.compile(
    r"pick\s+(?:the\s+)?(?P<object>.+?)\s+and\s+place\s+(?:it\s+)?"
    r"(?:(?:on|onto|in|into|to)\s+(?:the\s+)?)?(?P<target>.+)$",
    re.IGNORECASE,
)
CHINESE_PICK_PLACE = re.compile(r"把(?P<object>.+?)放到(?P<target>.+?)(?:上|里|中)?$")


def parse_robot_command(text: str) -> RobotCommand:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        raise ValueError("Command text cannot be empty.")

    english_match = ENGLISH_PICK_PLACE.search(cleaned)
    if english_match:
        return RobotCommand(
            action="pick_place",
            object=english_match.group("object").strip(),
            target=english_match.group("target").strip(),
        )

    chinese_match = CHINESE_PICK_PLACE.search(cleaned)
    if chinese_match:
        return RobotCommand(
            action="pick_place",
            object=chinese_match.group("object").strip(),
            target=chinese_match.group("target").strip(),
        )

    lower = cleaned.lower()
    if lower.startswith("pick "):
        return RobotCommand(action="pick", object=cleaned[5:].strip())
    if cleaned.startswith("抓取"):
        return RobotCommand(action="pick", object=cleaned[2:].strip())

    raise ValueError(f"Unsupported robot command: {text}")
