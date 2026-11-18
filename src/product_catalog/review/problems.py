"""The things a reviewer can tick, and the exact sentence each one adds to the prompt.

Free-text notes are vague ("looks wrong"), and an image model acts on the wording it
is given. Each box here is a fixed, unambiguous instruction, and whatever the
reviewer types is added after them.
"""

from __future__ import annotations

IMAGE_PROBLEMS = {
    "cropped": (
        "The product is cut off",
        "Show the whole product: pull the camera back so no part of it touches or "
        "crosses the edge of the image.",
    ),
    "text": (
        "Brand or text is wrong",
        "Copy the printed brand name and every word on the product letter for letter "
        "from the reference. Do not alter, misspell, translate or invent any text.",
    ),
    "extras": (
        "Extra people or objects",
        "Remove every person, animal, prop and object that is not the product itself "
        "or named in the setting.",
    ),
    "shape": (
        "Product does not match the photo",
        "Match the reference exactly: the same shape, colour, parts and proportions. "
        "Do not add, remove or redesign any part.",
    ),
    "scene": (
        "Background is wrong",
        "Make the background match the setting exactly and keep it clearly visible. "
        "Do not substitute a different place.",
    ),
}

DESCRIPTION_PROBLEMS = {
    "name": (
        "Brand or product name is wrong",
        "Use the exact product name and brand given in the title. Do not rename it.",
    ),
    "invented": (
        "Invents features or sizes",
        "Remove every claim, size, capacity or feature that is not visible in the "
        "photos or stated in the title.",
    ),
    "length": (
        "Too long or too short",
        "Write 40 to 70 words in one short paragraph.",
    ),
}

CATALOGS = {"image": IMAGE_PROBLEMS, "description": DESCRIPTION_PROBLEMS}


def compose_note(kind: str, ticked: list[str], text: str) -> str:
    """The ticked sentences in a fixed order, then the reviewer's own words.

    Unknown keys are ignored rather than trusted: the form is the only thing that
    should be able to put these sentences into a prompt.
    """
    catalog = CATALOGS[kind]
    sentences = [catalog[key][1] for key in catalog if key in ticked]
    if text.strip():
        sentences.append(text.strip())
    return " ".join(sentences)
