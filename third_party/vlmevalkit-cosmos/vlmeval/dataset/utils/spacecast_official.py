"""Verbatim excerpt of the official SpaceCast-Bench evaluator (do not edit by hand).

Source: https://github.com/ZJU-REAL/SpaceCast-Bench @ 6696e5eb41371a9c8038c0d902d2aa488b800593
  spacecast_eval/protocol.py: prompt constants, MODE_RULES, build_prompt, prepare_image, answer parsing
  spacecast_eval/data.py:     LEVELS, TYPE_NAMES, release counts, category, source_type, view_metadata
License: Apache-2.0 (upstream LICENSE). Each top-level definition below is copied byte-for-byte
from the files above (extracted with `ast` line spans), only reordered into one module.
Used by vlmeval/dataset/spacecast.py; parity with upstream is checked by
scripts/check_spacecast_parity.py in the Cosmos3-Reasoner repo.
"""
import io
import re
from collections import defaultdict

PROTOCOL = "spacecast-v1"


SYSTEM_PROMPT = (
    "You are a careful vision-language assistant solving multiple-choice "
    "spatial reasoning questions about an image."
)


PROMPT_SUFFIX = (
    "Work through your reasoning briefly, then give your final choice as the LAST line.\n"
    "Keep your reasoning as short as possible (a few sentences at most) so the final Answer line is not truncated.\n"
    "Return this format:\n"
    "Reasoning: <brief reasoning>\n"
    "Answer: <single letter>"
)


MULTI_SELECT_PROMPT_SUFFIX = (
    "Work through your reasoning briefly, then give your final choice as the LAST line.\n"
    "If more than one option is correct, list all letters comma-separated.\n"
    "Keep your reasoning as short as possible (a few sentences at most) so the final Answer line is not truncated.\n"
    "Return this format:\n"
    "Reasoning: <brief reasoning>\n"
    "Answer: <letter(s)>"
)


DIRECT_PROMPT_SUFFIX = "Answer with a single letter only (A, B, C, or D). Do not explain."


DIRECT_MULTI_SELECT_PROMPT_SUFFIX = (
    "Answer with the correct letter(s) only, comma-separated if more than one. Do not explain."
)


MODE_RULES = {
    "cot": {"suffix": PROMPT_SUFFIX, "multi_select_suffix": MULTI_SELECT_PROMPT_SUFFIX,
            # Reasoning returned in a separate field is kept, ahead of the answer.
            "include_reasoning": True,
            "retry_unparsed": False,
            "qwen35_enable_thinking": True},
    "direct": {"suffix": DIRECT_PROMPT_SUFFIX, "multi_select_suffix": DIRECT_MULTI_SELECT_PROMPT_SUFFIX,
               # Only the final answer content is used.
               "include_reasoning": False,
               # A response without a parseable option is requested again, and is
               # recorded as an error once the retries are used up.
               "retry_unparsed": True,
               "qwen35_enable_thinking": False},
}


IMAGE_MAX_PX = 1280


IMAGE_JPEG_QUALITY = 90


def check_mode(mode):
    if mode not in MODE_RULES:
        raise ValueError(f"Unknown evaluation mode {mode!r}; expected one of {', '.join(MODES)}")
    return mode


def build_prompt(row, mode):
    rules = MODE_RULES[check_mode(mode)]
    parts = [str(row.get("question") or "").strip(), ""]
    for idx, option in enumerate(row.get("options") or []):
        parts.append(f"{chr(65 + idx)}) {option}")
    parts.extend(["", rules["multi_select_suffix"] if row.get("multi_select") else rules["suffix"]])
    return "\n".join(parts)


def prepare_image(blob):
    from PIL import Image
    with Image.open(io.BytesIO(blob)) as image:
        image = image.convert("RGB")
        image.thumbnail((IMAGE_MAX_PX, IMAGE_MAX_PX))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=IMAGE_JPEG_QUALITY)
    return out.getvalue()


def parse_prediction(value, allowed, multi):
    """Validate an explicit prediction (a letter, a comma list or a list of letters)."""
    if isinstance(value, str):
        value = value.strip().upper()
        if not re.fullmatch(r"[A-Z](?:\s*[,;]\s*[A-Z])*", value):
            return None
        letters = re.split(r"\s*[,;]\s*", value)
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        letters = [v.strip().upper() for v in value]
    else:
        return None
    if (not letters or len(set(letters)) != len(letters) or
            not set(letters) <= set(allowed) or (not multi and len(letters) != 1)):
        return None
    return frozenset(letters)


def allowed_letters(row):
    n_options = len(row.get("options") or [])
    if n_options <= 0:
        return "ABCD"
    return "".join(chr(65 + idx) for idx in range(min(n_options, 26)))


def _ordered_unique_letters(values, letters):
    seen = {value.upper() for value in values if value and value.upper() in letters.upper()}
    return [letter for letter in letters.upper() if letter in seen]


def _parse_multi_candidate(candidate, letters):
    allowed = re.escape(letters.upper())
    tokens = re.findall(rf"(?<![A-Z0-9])([{allowed}])(?![A-Z0-9])", candidate)
    if tokens:
        return _ordered_unique_letters(tokens, letters)
    compact = re.sub(r"[\s,;/&+|\-]+", "", candidate)
    if compact and re.fullmatch(rf"[{allowed}]+", compact):
        return _ordered_unique_letters(list(compact), letters)
    return []


def parse_answers(raw, letters):
    if not raw:
        return []
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()

    parsed = _parse_multi_candidate(upper, letters)
    if parsed and re.fullmatch(rf"[\s{allowed},;/&+|\-]+", upper):
        return parsed

    tail = re.search(
        rf"(?:^|[\r\n.!?])\s*([\(\[]?[{allowed}]"
        rf"(?:\s*(?:[,;/&+|\-]|\bAND\b)\s*[{allowed}])*\s*[\)\]]?)\s*$",
        upper,
    )
    if tail:
        parsed = _parse_multi_candidate(tail.group(1), letters)
        if parsed:
            return parsed

    patterns = [
        r"(?:FINAL\s+)?ANSWER(?:S)?(?:\s+IS|\s+ARE)?\s*[:：-]?\s*([^\r\n]+)",
        r"(?:CHOICES?|OPTIONS?)(?:\s+IS|\s+ARE)?\s*[:：-]?\s*([^\r\n]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, upper))
        if matches:
            parsed = _parse_multi_candidate(matches[-1].group(1), letters)
            if parsed:
                return parsed
    return []


def _normalized_option_text(value):
    return re.sub(r"[\W_]+", " ", str(value or "").casefold()).strip()


def _parse_explicit_option_text(raw, options):
    if not options:
        return None
    option_keys = defaultdict(list)
    for index, option in enumerate(options):
        key = _normalized_option_text(option)
        if key:
            option_keys[key].append(chr(65 + index))

    stripped = raw.strip()
    last_line = next(
        (line.strip() for line in reversed(stripped.splitlines()) if line.strip()),
        "",
    )
    candidates = [stripped] if "\n" not in stripped and "\r" not in stripped else []
    patterns = [
        r"^(?:[-*]\s*)?\**(?:final\s+)?(?:answer|choice|option)\**"
        r"(?:\s+(?:is|would\s+be|should\s+be))?\s*[:：-]?\s*(.+?)\s*$",
        r"^(?:[-*]\s*)?\**(?:the|my)\s+(?:final\s+)?(?:answer|choice|option)\**"
        r"\s+is\s*[:：-]?\s*(.+?)\s*$",
        r"^(?:[-*]\s*)?(?:i(?:'ll|\s+will|\s+would)?\s+)?"
        r"(?:choose|select|go\s+with)\s*[:：-]?\s*(.+?)\s*$",
        r"^(?:[-*]\s*)?(?:so|therefore|thus|hence|in\s+conclusion)"
        r"\s*[:,]?\s*(.+?)\s*$",
        r"^(?:[-*]\s*)?(.+?)\s+is\s+(?:the|my)\s+(?:final\s+)?answer\s*[.!]?\s*$",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, last_line, flags=re.IGNORECASE)
        if match:
            candidates.append(match.group(1))

    matched_letters = set()
    for candidate in candidates:
        letters_for_key = option_keys.get(_normalized_option_text(candidate), [])
        if len(letters_for_key) == 1:
            matched_letters.add(letters_for_key[0])
    return next(iter(matched_letters)) if len(matched_letters) == 1 else None


def parse_answer(raw, letters, options=None):
    if not raw:
        return None
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()
    if re.fullmatch(rf"[{allowed}]", upper):
        return upper

    # Some models append the final letter to the last punctuation mark after
    # emitting reasoning (for example, "...sense.B").
    tail = re.search(rf"(?:^|[\r\n.!?])\s*[\(\[]?([{allowed}])[\)\].!?]?\s*$", upper)
    if tail:
        return tail.group(1)

    patterns = [
        rf"(?:FINAL\s+)?ANSWER(?:\s+IS)?\s*[:：-]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"(?:CHOICE|OPTION)(?:\s+IS)?\s*[:：-]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"^[\(\[]?\s*([{allowed}])\s*[\)\].:：-]",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, upper))
        if matches:
            return matches[-1].group(1)
    return _parse_explicit_option_text(raw, options)


def parse_response(text, row):
    """Sorted predicted letters, or None when no answer can be recovered. Same in both modes."""
    if not isinstance(text, str):
        return None
    letters = allowed_letters(row)
    if row.get("multi_select"):
        predictions = parse_answers(text, letters)
        return sorted(predictions) if predictions else None
    prediction = parse_answer(text, letters, row.get("options"))
    return [prediction] if prediction else None


# ---- spacecast_eval/data.py ----

LEVELS = {
    "direction_agent": "L1", "direction_allocentric": "L1",
    "direction_object_centric": "L1", "distance": "L1", "occlusion": "L1",
    "attachment_chain": "L1", "object_move_agent": "L2",
    "object_move_allocentric": "L2", "object_move_object_centric": "L2",
    "object_move_distance": "L2", "object_move_occlusion": "L2",
    "object_remove": "L2", "object_rotate_object_centric": "L2",
    "coordinate_rotation_agent": "L3", "coordinate_rotation_allocentric": "L3",
    "coordinate_rotation_object_centric": "L3",
}


TYPE_NAMES = {
    "direction_agent": "Camera Direction",
    "direction_object_centric": "Object Direction",
    "direction_allocentric": "World Direction",
    "distance": "Distance",
    "occlusion": "Occlusion",
    "attachment_chain": "Attachment",
    "object_move_agent": "Move Camera Direction",
    "object_move_object_centric": "Move Object Direction",
    "object_rotate_object_centric": "Rotate Object Direction",
    "object_move_allocentric": "Move World Direction",
    "object_move_distance": "Move Distance",
    "object_move_occlusion": "Move Occlusion",
    "object_remove": "Remove Occlusion",
    "coordinate_rotation_agent": "Rotate Camera Direction",
    "coordinate_rotation_object_centric": "Rotate Object Direction",
    "coordinate_rotation_allocentric": "Rotate World Direction",
}


RELEASE_QUESTIONS = 3862


RELEASE_SCENES = 182


RELEASE_CATEGORY_COUNTS = {
    "L1 / Attachment": 97, "L1 / Camera Direction": 264, "L1 / Distance": 272,
    "L1 / Object Direction": 251, "L1 / Occlusion": 221, "L1 / World Direction": 263,
    "L2 / Move Camera Direction": 293, "L2 / Move Distance": 285,
    "L2 / Move Object Direction": 256, "L2 / Move Occlusion": 139,
    "L2 / Move World Direction": 264, "L2 / Remove Occlusion": 216,
    "L2 / Rotate Object Direction": 256, "L3 / Rotate Camera Direction": 263,
    "L3 / Rotate Object Direction": 256, "L3 / Rotate World Direction": 266,
}


def category(row):
    """Scoring category; level is part of the key so same-name L2/L3 tasks stay apart."""
    return f"{row['level']} / {TYPE_NAMES[source_type(row)]}"


def source_type(row):
    """Resolve paper labels using level; L2/L3 Rotate Object Direction differ."""
    label = row["type"]
    code = row.get("source_type")
    if code is None:
        if label in LEVELS:
            code = label
        else:
            matches = [k for k, name in TYPE_NAMES.items() if name == label and LEVELS[k] == row["level"]]
            if len(matches) != 1:
                raise ValueError(f"Unknown or ambiguous type/level: {row['question_uid']}")
            code = matches[0]
    if (code not in LEVELS or LEVELS[code] != row["level"] or
            label not in {code, TYPE_NAMES[code]} or
            (row.get("question_type") is not None and row["question_type"] != label)):
        raise ValueError(f"Inconsistent type/level/source_type: {row['question_uid']}")
    return code


def view_metadata(names):
    """The ordered sequence is main A, zero or more bridges, then main B."""
    if not 1 <= len(names) <= 8:
        raise ValueError("Each question must have 1 to 8 images")
    return {
        "main_image_a_name": names[0] if len(names) > 1 else None,
        "main_image_b_name": names[-1] if len(names) > 1 else None,
        "bridge_view_names": names[1:-1],
        "image_roles": (["main_a"] + ["bridge"] * (len(names) - 2) + ["main_b"]
                        if len(names) > 1 else ["image"]),
    }
