EVALUATE_STATE_PROMPT = """Your task is to rigorously evaluate whether the selected triplet from the knowledge graph is useful for reasoning toward answering the given question.

SCORING GUIDELINES:
0.0-0.3: Irrelevant or no contribution.
0.4-0.6: Somewhat relevant.
0.7-0.8: Relevant and helpful.
0.9-1.0: Directly answers or is critical.

OUTPUT FORMAT:
Provide only the numeric score between 0.0 and 1.0 with one decimal place on the first line. On the second line, include a concise one-sentence explanation.

Q: {question}
T: {triple}
RATING [0.0-1.0]:"""

entity_p_prompt = """Based on the question and path history, filter the most relevant {top_k} entities from the candidate entities.
Question: {question}
Current Entity: {current_entities}
Current Relation: {current_relation}
Path History: {path_history}
Candidate Entities: {candidate_names}

Please output the entity names in descending order of relevance, separated by commas:"""


