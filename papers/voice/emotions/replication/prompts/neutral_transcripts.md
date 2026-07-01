Write exactly {n_stories} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:

- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Output only the dialogue blocks. Do not include introductions, summaries, notes, explanations, "Step" headings, revised responses, or any text before the first dialogue or after the final dialogue.

Each dialogue must begin with a bracketed heading on its own line, numbered sequentially:

[dialogue 1]

[dialogue 2]

[dialogue 3]

Continue this exact pattern through [dialogue {n_stories}].

Inside each dialogue, use this format:

[dialogue N]

[optional untagged system instruction]

Person: [line]

AI: [line]

Person: [line]

AI: [line]

[continue for 2-6 exchanges]

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.

IMPORTANT: Do not use "System:" as a tag. If a dialogue includes system instructions, write them as one or two plain instruction sentences immediately after the [dialogue N] heading and before the first Person turn.

IMPORTANT: Do not use unbracketed headings such as "Dialogue 1", "Step 1", "Find a meeting location", or numbered list items. The only dialogue separator allowed is [dialogue N].

IMPORTANT: After [dialogue {n_stories}], finish that dialogue and stop. Do not add a note, recap, checklist, revised version, or confirmation that the task is complete.

Generate a diverse mix of dialogue types across the {n_stories} examples:

- Some, but not all should include a system prompt at the start. These should come before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks (brainstorming names, generating lists)
- If it's natural to do so given the topic, it's ok for the dialogue to be a single back and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.

- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion
