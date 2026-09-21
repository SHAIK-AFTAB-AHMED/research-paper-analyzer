def generate_detailed_summary(input_text, llm_model):
    """
    Generate a detailed and structured summary of research-paper text
    using the configured LLM.
    """

    if not input_text or not str(input_text).strip():
        return "No content is available to summarize."

    prompt = f"""
You are an expert research analyst and technical writer.

Your task is to carefully read the following research-paper text and
generate a comprehensive, structured summary covering the important
ideas, concepts, findings, arguments, examples, and data.

Do not provide a preamble.

Instructions:

1. Provide a structured summary including:
   - Main idea or theme of the text
   - Important subtopics or sections
   - Key findings, facts, or arguments
   - Important examples or data mentioned

2. Explain complex terms or concepts in a simple and intuitive way,
   as if teaching someone who is new to the topic.

3. Ensure clarity and depth.
   Avoid vague or generic statements.

4. Present the output using:
   - Clear headings
   - Bullet points
   - Short paragraphs

5. If the text is technical or academic, include a section titled:

   "Explanation in Simple Terms"

Input Text:

{input_text}

Output Format:

Title or Theme

Summary
- Well-structured explanation

Key Points
- Important point
- Important point
- Important point

Explanation in Simple Terms
- Simple explanation of the concepts
"""

    try:
        response = llm_model.invoke(prompt)

        if isinstance(response, str):
            return response.strip()

        if hasattr(response, "content"):
            return response.content.strip()

        if hasattr(response, "text"):
            return response.text.strip()

        return str(response).strip()

    except Exception as error:
        print("Error while generating summary:", error)
        return "Unable to generate the summary at this time."