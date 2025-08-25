Zero-shot prompt://
You are an expert in clinical natural language processing.

The task is to  the correct relationship between the following two entities found in the same sentence.

The Task:
Sentence: {note}
Entities: "{ent1}" ({label1}), "{ent2}" ({label2})

Choose only one of the following relations: suggestive_of, located_at, modify
Just output the chosen relation as a single word (no explanation).

Instructional prompt://
You are an expert in clinical natural language processing.
Based on the information provided, determine the relationship between two entities in a radiology sentence.

Here are the definitions of the entities and possible relationships:

Anatomy: Refers to anatomical body parts found in radiology reports, such as "lung".
Observation: Refers to findings or diagnoses from a radiology image, such as "effusion", "increased", or "clear".
Possible relationships:

suggestive_of (Observation, Observation): One observation implies or suggests another observation.
located_at (Observation, Anatomy): An observation is located at or associated with an anatomical body part.
modify (Observation, Observation) or (Anatomy, Anatomy): One entity modifies or quantifies the other.
Now classify the relation between the following two entities found in the same sentence:

What is the correct relationship between "{ent1}" ({label1}) and "{ent2}" ({label2}), based on their use in the same radiology sentence below?
"{note}"

Choose only one of the following relations: suggestive_of, located_at, modify
Just output the chosen relation as a single word (no explanation)

Chain-of-thought prompt://
You are an expert in clinical natural language processing.
Based on the information provided, determine the relationship between two entities in a radiology sentence.
Here are the definitions of the entities and possible relationships:
Anatomy: Refers to anatomical body parts found in radiology reports, such as "lung".
Observation: Refers to findings or diagnoses from a radiology image, such as "effusion", "increased", or "clear".
Possible relationships:
suggestive_of (Only between Observation and Observation): One observation implies or suggests another observation.
located_at (Only between Observation and Anatomy): An observation is located at or associated with an anatomical body part.
modify (between Observation and Observation) or (between Anatomy and Anatomy): One entity modifies or quantifies the other.
Think and run step by step:
1.First identify the type of entities, and eliminate impossible relationships.
2.Then choose only one of the following relations: suggestive_of, located_at, modify
3.Just output the chosen relationship as a single word (suggestive_of, located_at, modify, without any explanation or other words).
Q: What is the relationship between the two entities in this sentence?
Sentence: "There is a pleural effusion in the left lung."
Entities: "effusion" (Observation), "lung" (Anatomy)
Method: First identify the type of entity.  The effusion is an observation and lung is an Anatomy, so only located_at is possible. The effusion is described as being in the lung, which is an anatomical location. So this is a case of an observation occurring at an anatomy. The relationship is located_at.
Answer: located_at
Q: What is the relationship between the two entities in this sentence?
Sentence: "Severe scoliosis is causing asymmetry of the ribcage."
Entities: "scoliosis" (Observation), "asymmetry" (Observation)
Method: First identify the type of entity. The scoliosis is an observation and asymmetry is an Observation, so only suggestive_of is possible. Scoliosis is said to cause asymmetry. One observation implies the other. The relationship is suggestive_of.
Answer: suggestive_of
Q: What is the relationship between the two entities in this sentence?
Sentence: "A pre-existing left basal parenchymal opacity has completely cleared."
Entities: "left" (Anatomy), "basal" (Anatomy)
Method: First identify the type of entity. The "left" is an Anatomy and "basal" is an Anatomy, so only modify is possible. 'Left' refers to the position of 'basal', and both are anatomy terms. One anatomical description modifies the other. The relationship is modify.
Answer: modify
Q: What is the correct relationship between "{ent1}" ({label1}) and "{ent2}" ({label2}), based on their use in the same radiology sentence below?
"{note}"
Answer:(ONE WORD ONLY)

Few-shot prompt://
Based on the information provided, determine the relationship between two entities in a radiology sentence.

Here are some examples:

Sentence: "There is a pleural effusion in the left lung."
Entities: "effusion" (Observation), "lung" (Anatomy)
Answer:located_at.

Sentence: "Severe scoliosis is causing asymmetry of the ribcage."
Entities: "scoliosis" (Observation), "asymmetry" (Observation)
Answer:suggestive_of.

Sentence: "A pre-existing left basal parenchymal opacity has completely cleared."
Entities: "left" (Anatomy), "basal" (Anatomy)
Answer:modify.

Choose only one of the following relations: suggestive_of, located_at, modify
Just output the chosen relation as a single word (no explanation).

Now classify the relation between the following two entities found in the same sentence:
What is the correct relationship between "{ent1}" ({label1}) and "{ent2}" ({label2}), based on their use in the same radiology sentence below?
"{note}"


Instructional&few-shot prompt://
You are an expert in clinical natural language processing.
Based on the information provided, determine the relationship between two entities in a radiology sentence.

Here are the definitions of the entities and possible relationships:

Anatomy: Refers to anatomical body parts found in radiology reports, such as "lung".
Observation: Refers to findings or diagnoses from a radiology image, such as "effusion", "increased", or "clear".

Possible relationships:

suggestive_of (Observation, Observation): One observation implies or suggests another observation.
located_at (Observation, Anatomy): An observation is located at or associated with an anatomical body part.
modify (Observation, Observation) or (Anatomy, Anatomy): One entity modifies or quantifies the other.

Here are some examples:

Sentence: "There is a pleural effusion in the left lung."
Entities: "effusion" (Observation), "lung" (Anatomy)
Answer:located_at.

Sentence: "Severe scoliosis is causing asymmetry of the ribcage."
Entities: "scoliosis" (Observation), "asymmetry" (Observation)
Answer:suggestive_of.

Sentence: "A pre-existing left basal parenchymal opacity has completely cleared."
Entities: "left" (Anatomy), "basal" (Anatomy)
Answer:modify.

Choose only one of the following relations: suggestive_of, located_at, modify
Just output the chosen relation as a single word (no explanation).

Now classify the relation between the following two entities found in the same sentence:
What is the correct relationship between "{ent1}" ({label1}) and "{ent2}" ({label2}), based on their use in the same radiology sentence below?
"{note}"

