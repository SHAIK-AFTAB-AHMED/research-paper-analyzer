import json
from typing import List, Dict


def refine_sections(input_list: List[Dict], llm) -> List[Dict]:
    """
    Refine the sections detected from a research paper using an LLM.

    Removes:
    - Figure captions
    - Table captions
    - Incomplete fragments
    - Irrelevant section entries

    Returns a clean list of section dictionaries.
    """

    # Convert Python list into proper JSON
    input_json = json.dumps(
        input_list,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
You are a precise data processor.

I will give you a JSON list of sections from a research paper.

Each item may contain:
- "section" (string)
- optional "subsection" (string)
- "start" (integer)

Your task is to refine this list.

Remove completely:

1. Figure or Table captions.
   Any section beginning with "Figure" or "Table".

2. Incomplete, meaningless, or fragment sections.
   Examples:
   - "making"
   - "length is smaller"
   - incomplete sentences
   - random fragments

3. Any irrelevant entries that are not proper research-paper sections
   or subsections.

Keep only meaningful main sections and subsections.

Main section format:

{{
    "section": "Section Name",
    "start": number
}}

Subsection format:

{{
    "section": "Parent Section",
    "subsection": "Subsection Name",
    "start": number
}}

Important requirements:

- Return ONLY a valid JSON array.
- Do not include Markdown.
- Do not use ```json.
- Do not include explanations.
- Do not include comments.
- Do not include a preamble.
- Do not add information that is not present in the input.
- Preserve the original start positions.
- Return an empty JSON array [] if no valid sections remain.

Input JSON:

{input_json}

Return only the JSON array.
"""

    try:
        response = llm.invoke(prompt)

        # Extract text from LangChain response
        if hasattr(response, "content"):
            assistant_text = response.content.strip()
        else:
            assistant_text = str(response).strip()

        # Remove accidental Markdown code fences
        if assistant_text.startswith("```"):
            lines = assistant_text.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            assistant_text = "\n".join(lines).strip()

        # Parse JSON
        sections = json.loads(assistant_text)

        # Make sure the LLM actually returned a list
        if not isinstance(sections, list):
            print("Warning: LLM returned something other than a JSON list.")
            return []

        return sections

    except json.JSONDecodeError as error:
        print("Warning: LLM output was not valid JSON.")
        print("JSON error:", error)
        print("Raw LLM output:", assistant_text if "assistant_text" in locals() else "No output")
        return []

    except Exception as error:
        print("Error while refining sections:", error)
        return []


def split_sections_with_content(
    text: str,
    detected_sections: List[Dict]
) -> Dict[str, str]:
    """
    Split research paper text into sections/subsections using
    detected start positions.

    Returns a dictionary:

    {
        "Introduction": "...",
        "3.1 Methodology": "...",
        "Results": "..."
    }
    """

    if not text or not text.strip():
        return {"Full_Paper": ""}

    if not detected_sections:
        return {"Full_Paper": text}

    # Keep only valid section entries
    valid_sections = []

    for section in detected_sections:
        if not isinstance(section, dict):
            continue

        if "start" not in section:
            continue

        if "section" not in section:
            continue

        try:
            section["start"] = int(section["start"])
        except (TypeError, ValueError):
            continue

        valid_sections.append(section)

    if not valid_sections:
        return {"Full_Paper": text}

    # Sort according to starting position
    valid_sections = sorted(
        valid_sections,
        key=lambda x: x["start"]
    )

    results = {}

    for i, section in enumerate(valid_sections):

        start = section["start"]

        if i + 1 < len(valid_sections):
            end = valid_sections[i + 1]["start"]
        else:
            end = len(text)

        # Protect against invalid positions
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        section_text = text[start:end].strip()

        section_name = str(section["section"]).strip()

        subsection_name = section.get("subsection")

        if subsection_name:
            subsection_name = str(subsection_name).strip()

            if subsection_name:
                results[subsection_name] = section_text
        else:
            if section_name:
                results[section_name] = section_text

    return results