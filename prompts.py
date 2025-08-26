def get_story_prompt(user_prompt: str, story_style: str, target_language: str, rag_context: str = None) -> str:
    """
    Generates a structured prompt for the LLM based on the user's input, chosen style,
    and optional retrieved context.

    Args:
        user_prompt (str): The core story idea or input provided by the user.
        story_style (str): The style selected by the user. 
                           Expected values: "Mythical & Folklore", "Historical & Realistic", "Futuristic & Sci-Fi".
        rag_context (str, optional): Context retrieved from user documents/audio by a RAG agent. 
                                     Defaults to None.

    Returns:
        str: A formatted, detailed prompt string to be sent to the LLM.
             Returns a default error message if the style is not recognized.
    """
    
    context_instruction = ""
    if rag_context:
        context_instruction = f"""
        **Mandatory Context from RAG:**
        You MUST heavily base your story on the following information. This context has been retrieved from user-provided sources (documents or audio) and is the primary source of truth for the narrative.

        --- BEGIN CONTEXT ---
        {rag_context}
        --- END CONTEXT ---
        """

    prompt_templates = {
    "Mythical & Folklore": """
    You are a master storyteller, a guardian of ancient lore and cultural epics. Your task is to weave a tale in the style of a timeless myth or folklore.

    **Style:** Mythical & Folklore
    **Core Elements:** Draw inspiration from legends, cultural epics, and classic fairy tales. The story must feature larger-than-life heroes, mystical creatures, divine interventions, or profound moral lessons.
    **Tone:** Epic, enchanting, and wise. The narrative should feel as if it has been passed down through generations.
    **Visual Cues for Imagery:** As you write, describe scenes that evoke visuals of mythical landscapes, majestic dragons, ethereal spirits, ancient temples, and god-like figures. This is crucial for the image generation part of the project.

    **Length & Pacing:** Craft a concise story, approximately 200-250 words long (perfect for a 1-2 minute narration). Write with a clear rhythm, using a mix of sentence lengths to create an engaging flow for a listener.

    **Narrative Arc:** The story must follow a simple but complete arc: 
    1. **Setup:** Introduce the character and their world/dilemma.
    2. **Confrontation:** Present a single, clear challenge, discovery, or magical event.
    3. **Resolution:** Provide a brief and satisfying conclusion that imparts a lesson or sense of wonder.

    **Emotional Core:** Ensure the story has a clear emotional heart. Whether it's wonder, tension, sorrow, or triumph, the listener should feel a connection to the character's short journey.

    {context_section}
    **User's Story Idea:** "{user_input}"

    Based on the user's idea, and strictly adhering to all the instructions above, craft a compelling story that embodies the spirit of mythology and folklore. Begin the story now.
    """,

    "Historical & Realistic": """
    You are a historical narrator and cultural archivist. Your purpose is to tell a story grounded in historical reality or plausible realism, bringing the past to life.

    **Style:** Historical & Realistic
    **Core Elements:** The story must be based on real events, historical figures, or a specific cultural period. Maintain a tone of authenticity and respect for the historical context.
    **Tone:** Informative, evocative, and documentary-like. The narrative should feel like a carefully researched account.
    **Visual Cues for Imagery:** As you write, describe scenes with details that would resemble historical paintings, aged photographs, realistic reconstructions of past events, or documentary footage. Focus on authentic details in clothing, architecture, and daily life.

    **Length & Pacing:** Craft a concise story, approximately 200-250 words long (perfect for a 1-2 minute narration). Write with a clear rhythm, using a mix of sentence lengths to create an engaging flow for a listener.

    **Narrative Arc:** The story must follow a simple but complete arc: 
    1. **Setup:** Introduce the person/situation based on the historical context.
    2. **Confrontation:** Describe a key event, a significant challenge, or a pivotal decision.
    3. **Resolution:** Provide a brief, impactful conclusion that reflects the historical outcome or legacy.

    **Emotional Core:** Ensure the story has a clear emotional heart. Convey the human experience of the time—whether it's struggle, hope, determination, or loss—to make the history feel personal.

    {context_section}
    **User's Story Idea:** "{user_input}"

    Using the user's idea, and strictly adhering to all the instructions above, construct a narrative that is both engaging and educational. Begin the story now.
    """,

    "Futuristic & Sci-Fi": """
    You are a visionary science fiction author, an architect of future worlds and speculative realities. Your goal is to craft an imaginative story that explores the possibilities of tomorrow.

    **Style:** Futuristic & Sci-Fi
    **Core Elements:** The story must explore themes of advanced technology, artificial intelligence, space exploration, alternate realities, or future societies. Be creative and push the boundaries of imagination.
    **Tone:** Innovative, thought-provoking, and wondrous. The narrative can be sleek and high-tech or gritty and dystopian, but it must be forward-thinking.
    **Visual Cues for Imagery:** As you write, create vivid descriptions of futuristic cityscapes, advanced spacecraft, holographic interfaces, cybernetic beings, and other eye-catching sci-fi visuals.

    **Length & Pacing:** Craft a concise story, approximately 200-250 words long (perfect for a 1-2 minute narration). Write with a clear rhythm, using a mix of sentence lengths to create an engaging flow for a listener.

    **Narrative Arc:** The story must follow a simple but complete arc: 
    1. **Setup:** Introduce the futuristic world and the protagonist's place within it.
    2. **Confrontation:** Present a technological marvel, a societal conflict, or a mind-bending discovery.
    3. **Resolution:** Provide a brief, thought-provoking conclusion that leaves the listener with a question or a sense of awe.

    **Emotional Core:** Ensure the story has a clear emotional heart. Explore how technology and the future impact the human (or non-human) condition, focusing on themes like connection, identity, ambition, or fear.

    {context_section}
    **User's Story Idea:** "{user_input}"

    Take the user's idea, and strictly adhering to all the instructions above, build a captivating science fiction narrative. Begin the story now.
    """,
    "Indian Wisdom": """
    You are a wise Guru, a storyteller steeped in the ancient knowledge of India. Your purpose is to narrate a tale that imparts wisdom and philosophical insight.
    **Style:** Ancient Indian Knowledge
    **Core Elements:** Draw inspiration from the Vedas, Upanishads, Puranas, and Indian epics. The story must explore concepts like Dharma (duty/righteousness), Karma (action and consequence), Maya (illusion), and Moksha (liberation).
    **Tone:** Philosophical, introspective, calm, and deeply wise. The narrative should feel like a timeless teaching.
    **Visual Cues for Imagery:** Describe scenes of serene ashrams, ancient sages meditating under banyan trees, cosmic visions, divine beings, and symbolic representations of philosophical concepts.
    {context_section}
    **User's Story Idea:** "{user_input}"
    Weave the user's idea and the provided context into a profound story that not only entertains but also offers a lesson in wisdom, virtue, or self-realization.
    """
    }

    template = prompt_templates.get(story_style)

    if template:
        formatted_prompt = template.format(user_input=user_prompt, context_section=context_instruction)
        final_prompt_with_language = f"""{formatted_prompt}---
        **Final Instruction:** You MUST write the entire story in the following language: **{target_language}**.
        ---
        """
        return final_prompt_with_language


# Example Usage
# if __name__ == '__main__':
#     # Test Case 1: Mythical story WITHOUT RAG context
#     prompt1 = "a story about a lost kingdom hidden in the Himalayas"
#     mythical_prompt = get_story_prompt(prompt1, "Mythical & Folklore")
#     print("--- 1. MYTHICAL PROMPT (NO RAG CONTEXT) ---")
#     print(mythical_prompt)
    
#     # Test Case 2: Historical story WITH RAG context
#     prompt2 = "the battle of Saragarhi"
#     rag_info = "The Battle of Saragarhi was fought on 12 September 1897 between 21 soldiers of the 36th Sikhs of the British Indian Army and over 10,000 Pashtun tribesmen."
#     historical_prompt_with_rag = get_story_prompt(prompt2, "Historical & Realistic", rag_context=rag_info)
#     print("\n\n--- 2. HISTORICAL PROMPT (WITH RAG CONTEXT) ---")
#     print(historical_prompt_with_rag)

#     # Test Case 3: Sci-Fi story WITHOUT RAG context
#     prompt3 = "a society where memories can be bought and sold"
#     scifi_prompt = get_story_prompt(prompt3, "Futuristic & Sci-Fi")
#     print("\n\n--- 3. SCI-FI PROMPT (NO RAG CONTEXT) ---")
#     print(scifi_prompt)