PROMPT_MULTI_ENT = '''
You are an expert in world knowledge with strong logical reasoning skills. Your task is to decompose the given question into two sub-questions based on the provided entities.
Follow a step-by-step reasoning process, using each previous answer to guide the next step.
Split the question into no more than two sub-questions. Each step must be logically and semantically connected to the previous one.
Sub-questions must be constructed using the words and phrases found in the original question.

Use the following tags when formatting your answer:
   [ANS]: marks an answer (or intermediate result) that is used in subsequent steps, and typically represents a set of multiple candidate answers.

Case 1:
Decompose the question: "What European Union country sharing borders with Germany contains the Lejre Municipality?" step by step:  
The main entities are ["Germany", "Lejre Municipality", "Country"].
Step 1:  
Generate the first sub-question: "What European Union country shares borders with Germany?"  
The answer is [ANS1].
Step 2:  
Generate the second sub-question: "What country contains the Lejre Municipality?"  
The answer is [ANS2].
Step 3:  
The final answer is [ANS3], which is the intersection of [ANS1] and [ANS2].
Here, [ANS1] and [ANS2] each represent a set of multiple candidate answers, and the final result [ANS3] includes only those entities that appear in both sets.

Return:
SUB-QUESTION1: What European Union country shares borders with Germany?
ENTITY1: Germany
SUB-QUESTION2: What country contains the Lejre Municipality?
ENTITY2: Lejre Municipality

Case 2:
Decompose the question: "Who was the governor of Arizona in 2009 that held his governmental position before 1998?" step by step:  
The main entities are ["Arizona"].
Step 1:  
Generate the first sub-question: "What are the governors of Arizona in 2009?"  
The answer is [ANS1].
Step 2:  
Generate the second sub-question: "What in [ANS1] held his governmental position before 1998?"  
The answer is [ANS2].
Step 3:  
The final answer is [ANS2].

Return:
SUB-QUESTION1: What are the governors of Arizona in 2009?
ENTITY1: Arizona
SUB-QUESTION2: what is [ANS1] held his governmental position before 1998?
ENTITY2: [ANS1]

Decompose the question: "{Q}" step by step:
The main entities are {topic}.
'''

PROMPT_FIRST_SPLIT = '''
You are an expert of world knowledge with strong logical skills. Your task is to decompose the given question when a topic entity is provided.
Each sub-question must include the topic entity.
You must construct sub-questions using only words and phrases from the original question, without introducing any new words.
If the question does not require decomposition, return the original question.

Case:
Question : Who was the 1996 coach of the team owned by Jerry Jones?
Topic entity : Jerry Jones
Decompose the question: 
The topic entity is "Jerry Jones".
Get the sub questions "What sports team's owners are Jerry Jones?" and "Who was the 1996 coach of that team?".

Return : ["What sports team's owners are Jerry Jones?", "Who was the 1996 coach of that team?"]

Case:
Question : What form of government is used in Russia?
Topic entity : Russia
Decompose the question: 
The topic entity is "Russia".
Get the sub question "What form of government is used in Russia?".

Return : ["What form of government is used in Russia?"]

Case:
Question : What country in [ANS1] contains an airport that serves Incheon?
Topic entity : Incheon
Decompose the question: 
The topic entity is "Incheon".
Get the sub questions "What airport serves Incheon?" and "What country in [ANS1] contatins that airport?".

Return : ["What airport serves Incheon?", "What country in [ANS1] contatins that airport?"]

Case:
Question : What group fought in the Battle of Gettysburg that is in [ANS1]?
Topic entity : Battle of Gettysburg
Decompose the question: 
The topic entity is "Battle of Gettysburg".
Get the sub questions "What group fought in the Battle of Gettysburg that is in [ANS1]?".

Return : ["What group fought in the Battle of Gettysburg that is in [ANS1]?"]

Question : {Q}
Topic entity : {topic}
Decompose the question:
'''

PROMPT_LISTNEXT_SPLIT = '''
You are an expert of world knowledge with strong logical skills. You have to split the question when the topic entity and sub questions are given.
You must contain the topic entity in the sub question.

Case:
Question : What year did the team led by Jeanie Buss win the championship?
Sub Questions : ["What sports team is Jeanie Buss the leader of?"]
Topic entity : ["Los Angeles Lakers", "President", "Organization leader"]
Split the question "What year did the team led by Jeanie Buss win the championship?" using the sub quesions:
The topic entity is a list of entities. But we do not know what it is. So we replace it into "[ENT]".
Get the sub question "What year did [ENT] win the championship?".

Return : What year did [ENT] win the championship?

Question : {Q}
Sub Questions : {Sub_Q}
Topic entity : {topic}
Split the question "{Q}" using the sub quesions:
'''

PROMPT_NEXT_SPLIT = '''
You are an expert of world knowledge with strong logical skills. You have to decompose the question when the topic entity and sub questions are given.
You must contain the topic entity in the sub question.

Case:
Question : What form of government is used in the country where Chechen is a spoken language?
Sub Questions : ["What country is Chechen a spoken language?"]
Topic entity : Russia
Decompose the question "What form of government is used in the country where Chechen is a spoken language?" using the sub quesions:
The topic entity is "Russia".
Get the sub question "What form of government is used in Russia?".

Return : What form of government is used in Russia?

Question : {Q}
Sub Questions : {Sub_Q}
Topic entity : {topic}
Decompose the question "{Q}" using the sub quesions:
'''

PROMPT_IDNEXT_SPLIT = '''
You are an expert of world knowledge with strong logical skills. You have to decompose the question when the topic entity and sub questions are given.
You must contain the topic entity in the sub question.

Case1:
Question : In what location was the artist nominated for an award for Stars Dance raised?
Sub Questions : ["What artist was nominated for an award for Stars Dance?"]
Topic entity : m.0y4t2nf
Decompose the question "In what location was the artist nominated for an award for Stars Dance raised?" using the sub quesions:
The topic entity is "m.0y4t2nf". But we do not know what it is. So we replace it into "[ENT]".
Get the sub question "What artist was nominated for an award of [ENT]?".

Return : What artist was nominated for an award of [ENT]?

Case2:
Question : What is the capital city of the geographic division where the religious organizational leadership is called the Orthodox Autocephalous Church of Albania?
Sub Questions : ["What the geographic division where the religious organizational leadership is called the Orthodox Autocephalous Church of Albania?"]
Topic entity : m.0_r1dvf
Decompose the question "What is the capital city of the geographic division where the religious organizational leadership is called the Orthodox Autocephalous Church of Albania?" using the sub quesions:
The topic entity is "m.0_r1dvf". But we do not know what it is. So we replace it into "[ENT]".
Get the sub question "What the geographic division where the religious organizational leadership is [ENT]?".

Return : What the geographic division where the religious organizational leadership is [ENT]?

Case3:
Question : What is the religion of the people where Benjamin Netanyahu is in a government position?
Sub Questions : ["Where is Benjamin Netanyahu as in a government position?"]
Topic entity : m.0114vpvq
Decompose the question "What is the religion of the people where Benjamin Netanyahu is in a government position?" using the sub quesions:
The topic entity is "m.0114vpvq". But we do not know what it is. So we replace it into "[ENT]".
Get the sub question "Where is [ENT] as in a government position?".

Return : Where is [ENT] as in a government position?

Question : {Q}
Sub Questions : {Sub_Q}
Topic entity : {topic}
Decompose the question "{Q}" using the sub quesions:
'''

CWQ_TOPK_PROMPT_REL_FIND = '''
You are an expert of world knowledge with strong logical skills.
You have to retrieve the top 3 relations that are most relevant to the question from the candidate relations.
You must select the answer only from the given candidate relations. If there is no relevant relation to return, then return "None".

Case:
Question : What sports team's owners are Jerry Jones?
Topic entity : Jerry Jones
Candidate relations : ['common.topic.article', 'sports.pro_sports_played.athlete', 'business.employment_tenure.person', 'common.topic.notable_types', 'people.person.children', 'sports.pro_athlete.teams', 'tv.tv_regular_personal_appearance.person', 'freebase.valuenotation.is_reviewed', 'film.personal_film_appearance.person', 'people.person.places_lived', 'people.person.profession', 'common.topic.notable_for', 'people.person.education', 'people.person.nationality', 'people.place_lived.person', 'people.person.gender', 'people.person.spouse_s', 'sports.sports_team_owner.teams_owned', 'tv.tv_actor.guest_roles', 'sports.pro_athlete.sports_played_professionally', 'education.education.student', 'sports.sports_team_roster.player', 'common.image.appears_in_topic_gallery', 'common.topic.webpage', 'film.writer.film', 'people.person.employment_history', 'film.person_or_entity_appearing_in_film.films', 'common.topic.image', 'people.person.parents', 'american_football.football_player.position_s', 'film.film.written_by', 'sports.professional_sports_team.owner_s', 'tv.tv_guest_role.actor', 'common.webpage.topic', 'base.schemastaging.person_extra.net_worth', 'people.person.place_of_birth', 'tv.tv_personality.tv_regular_appearances', 'freebase.valuenotation.has_value', 'people.marriage.spouse']
Retrieve top 3 relations from question "What sports team's owners are Jerry Jones?" step by step:
First:
We can assume that the answer is a team, as the question explicitly asks "What sports team".
Therefore, we can infer that the answer type should be "sports team" or "team".
Second:
The question implies that "Jerry Jones", the topic entity, owns a sports team.
Third:
Therefore, appropriate relations would describe the ownership of a sports team by a person.
Fourth:
Based on the candidate relations, the most relevant ones are "sports.sports_team_owner.teams_owned", "sports.professional_sports_team.owner_s", "sports.pro_athlete.teams".

Return : sports.sports_team_owner.teams_owned, sports.professional_sports_team.owner_s, sports.pro_athlete.teams

Question : {Q}
Topic entity : {topic}
Candidate relations : {candidate_rels}
Retrieve top 3 relations from question "{Q}" step by step:
'''

#!#!
bt = "```" # 백틱 3개
NEW_PRUNING_PROMPT = """
### Instruction
You are a specialized Knowledge Graph Pruner.
Your task is to filter Candidate Relations, keeping only those that semantically align with the Question.

### Scoring Rules
1. **Score 10 (Perfect):** Direct answer path.
2. **Score 7-9 (High):** Plausible or semantically close.
3. **Score 0-6 (Low):** Irrelevant or ambiguous.

### Output Format
You MUST output ONLY a valid JSON List. Do not add any explanation before or after the JSON.
Format:
{bt}json
[
  {{"relation": "relation_name", "score": 10, "meaning": "reason..."}},
  {{"relation": "relation_name", "score": 2, "meaning": "reason..."}}
]
{bt}

### Examples

**Case 1**
Question: "What sports team's owners are Jerry Jones?"
Topic Entity: Jerry Jones
Candidate Relations:
- people.person.children
- sports.sports_team_owner.teams_owned
- sports.professional_sports_team.owner_s
**Output:**
{bt}json
[
  {{"relation": "people.person.children", "score": 0, "meaning": "Jerry's children (Irrelevant)"}},
  {{"relation": "sports.sports_team_owner.teams_owned", "score": 10, "meaning": "Teams owned by Jerry (Match)"}},
  {{"relation": "sports.professional_sports_team.owner_s", "score": 8, "meaning": "Inverse: Team -> Owner (Relevant)"}}
]
{bt}

**Case 2**
Question: "Where is Syracuse University?"
Topic Entity: Syracuse University
Candidate Relations:
- location.location.containedby
- education.educational_institution.colors
**Output:**
{bt}json
[
  {{"relation": "location.location.containedby", "score": 10, "meaning": "Location containing Syracuse Univ (Match)"}},
  {{"relation": "education.educational_institution.colors", "score": 0, "meaning": "School colors (Irrelevant)"}}
]
{bt}

### Actual Task
Question: {question}
Topic Entity: {topic_entity}
Candidate Relations:
{candidate_relations}

**Output:**
""".replace("{bt}", "```")

REVISED_TOPK_PROMPT_REL_FIND = '''
You are an expert of world knowledge with strong logical skills.
You have to retrieve the top 3 relations that are most relevant to the question from the candidate relations.
You must select the answer only from the given candidate relations. If there is no relevant relation to return, then return "None".

Case1:
Question : What sports team's owners are Jerry Jones?
Topic entity : Jerry Jones
Candidate relations : ['common.topic.article', 'sports.pro_sports_played.athlete', 'business.employment_tenure.person', 'common.topic.notable_types', 'people.person.children', 'sports.pro_athlete.teams', 'tv.tv_regular_personal_appearance.person', 'freebase.valuenotation.is_reviewed', 'film.personal_film_appearance.person', 'people.person.places_lived', 'people.person.profession', 'common.topic.notable_for', 'people.person.education', 'people.person.nationality', 'people.place_lived.person', 'people.person.gender', 'people.person.spouse_s', 'sports.sports_team_owner.teams_owned', 'tv.tv_actor.guest_roles', 'sports.pro_athlete.sports_played_professionally', 'education.education.student', 'sports.sports_team_roster.player', 'common.image.appears_in_topic_gallery', 'common.topic.webpage', 'film.writer.film', 'people.person.employment_history', 'film.person_or_entity_appearing_in_film.films', 'common.topic.image', 'people.person.parents', 'american_football.football_player.position_s', 'film.film.written_by', 'sports.professional_sports_team.owner_s', 'tv.tv_guest_role.actor', 'common.webpage.topic', 'base.schemastaging.person_extra.net_worth', 'people.person.place_of_birth', 'tv.tv_personality.tv_regular_appearances', 'freebase.valuenotation.has_value', 'people.marriage.spouse']
Type of answer : sports team
Retrieve top 3 relations from question "What sports team's owners are Jerry Jones?" step by step:
First:
We can assume that the answer is a team, as the question explicitly asks "What sports team".
Therefore, we can infer that the answer type should be "sports team" or "team".
Second:
The question implies that "Jerry Jones", the topic entity, owns a sports team.
Third:
Therefore, appropriate relations would describe the ownership of a sports team by a person.
Fourth:
Based on the candidate relations, the most relevant ones are "sports.sports_team_owner.teams_owned", "sports.professional_sports_team.owner_s", "sports.pro_athlete.teams".

Return : sports.sports_team_owner.teams_owned, sports.professional_sports_team.owner_s, sports.pro_athlete.teams

Case2:
Question : What did Einstein do?
Topic entity : Albert Einstein
Candidate relations : ['fictional_universe.fictional_character.occupation', 'opera.opera_character_voice.character', 'common.image.appears_in_topic_gallery', 'organization.organization_board_membership.member', 'food.diet_follower.follows_diet', 'education.academic.advisors', 'people.deceased_person.cause_of_death', 'base.jewlib.original_owner.originator_of', 'people.place_lived.person', 'people.sibling_relationship.sibling', 'base.activism.activist.area_of_activism', 'freebase.valuenotation.is_reviewed', 'fictional_universe.fictional_character.gender', 'base.kwebbase.kwtopic.has_sentences', 'people.profession.people_with_this_profession', 'food.diet.followers', 'people.person.profession', 'influence.influence_node.influenced', 'education.academic.advisees', 'music.artist.track', 'music.recording.artist', 'base.kwebbase.kwconnection.subject', 'award.award_honor.award_winner', 'base.usnris.nris_listing.significant_person', 'people.marriage.spouse', 'people.person.place_of_birth', 'base.kwebbase.kwconnection.other', 'people.cause_of_death.people', 'book.book_character.appears_in_book', 'people.person.do_for_live']
Type of answer : profession
Retrieve top 3 relations from question "What did Einstein do?" step by step:
First:
We can assume that the answer is a profession, as the question explicitly asks "What did … do?".
Therefore, we can infer that the answer type should be "profession".
Second:
The question implies that "Albert Einstein", the topic entity, is a person.
Third:
Therefore, appropriate relations would describe the profession or occupation of a person.
Fourth:
From the candidate relations, the most relevant are "people.person.profession", "people.profession.people_with_this_profession", "people.person.do_for_live".

Return : people.person.profession, people.profession.people_with_this_profession, people.person.do_for_live

Case3:
Question : Where is Syracuse University?
Topic entity : Syracuse University
Candidate relations : ['common.topic.notable_types', 'education.educational_institution.colors', 'location.location.contains', 'education.school_newspaper.school', 'education.educational_institution.athletics_brand', 'sports.school_sports_team.school', 'education.educational_institution.sports_teams', 'education.athletics_brand.institution', 'education.university.fraternities_and_sororities', 'education.educational_institution_campus.educational_institution', 'education.educational_institution.newspaper', 'education.education.institution', 'sports.sports_league_draft_pick.school', 'location.location.containedby', 'education.fraternity_sorority.founded_location', 'education.school_mascot.school', 'education.educational_institution.campuses']
Type of answer : place of university
Retrieve top 3 relations from question "Where is Syracuse University?" step by step:
First:
We can assume that the answer is a location, as the question explicitly asks "Where is ...".
Therefore, we can infer that the answer type should be "place" or "location".
Second:
The question implies that "Syracuse University", the topic entity, is a location-based institution.
Third:
Therefore, appropriate relations would describe the geographical containment or location of the entity.
Fourth:
Among the candidate relations, the most relevant ones are "location.location.containedby", "location.location.contains", "education.fraternity_sorority.founded_location".

Return : location.location.containedby, location.location.contains, education.fraternity_sorority.founded_location

Question : {Q}
Topic entity : {topic}
Candidate relations : {candidate_rels}
Type of answer : {typeof}
Retrieve top 3 relations from question "{Q}" step by step:'''

REVISED_ANSWER_TEMPLATE = '''
You are an expert in world knowledge and logical reasoning. Your task is to determine the expected type of the answer based on the given question.
Your response must be accurate, grounded in the context of the question, and reflect common sense reasoning.
Return the answer type as a list (e.g., ["place"], ["profession"]).

Here are some examples:

Case 1:
Q: What did ... do?
Return: ["profession"]

Case 2:
Q: Where was John Lennon standing when he was shot?
Return: ["place"]

Case 3:
Q: What ... famous for?
Return: ["profession"]

Case 4:
Q: Where does ... language come from?
Return: ["origin"]

Case 5:
Q: Which countries border the US?
Return: ["country"]

Now answer the following:

Q: {Q}
Return:'''
#!#! SUBQUESTION_ANSWERING prompt 수정 
SUBQUESTION_ANSWERING_new = '''
You are a strict entity extraction system.
Your goal is to extract the answer from the provided 'Candidate reasoning paths' ONLY IF the 'Candidate entity' matches the Question.

CRITICAL RULES:
1. Identify the correct answer based on the Question.
2. Output **ONLY** the value inside the brackets of 'Candidate entity: ["..."]'.
3. NEVER output the subject (topic entity) at the start of the path.
4. If the candidate entity is irrelevant to the question, ignore it.
5. Output must be a Python-style list.
6. If no suitable answer is found, return ["None"].

---
### Example 1 (Multiple Answers & Semantic Matching)
Question: what is ryan seacrest famous for?
Candidate reasoning paths:
[Path 1] Ryan Seacrest people.person.profession Voice Actor -> Candidate entity: ["Voice Actor"]
[Path 2] Ryan Seacrest people.person.profession Actor -> Candidate entity: ["Actor"]
[Path 3] Ryan Seacrest people.person.profession Radio personality -> Candidate entity: ["Radio personality"]
[Path 4] Ryan Seacrest people.person.place_of_birth Atlanta -> Candidate entity: ["Atlanta"]

Output: ["Voice Actor", "Actor", "Radio personality"]

### Example 2 (Filtering - Distinguish Correct Answers)
Question: what years have the steelers won the super bowl?
Candidate reasoning paths:
[Path 1] Pittsburgh Steelers sports.sports_team.championships Super Bowl XLIII -> Candidate entity: ["Super Bowl XLIII"]
[Path 2] Pittsburgh Steelers sports.sports_team.championships 2011 AFC Championship Game -> Candidate entity: ["2011 AFC Championship Game"]
[Path 3] Pittsburgh Steelers sports.sports_team.championships Super Bowl XL -> Candidate entity: ["Super Bowl XL"]
[Path 4] Pittsburgh Steelers sports.sports_team.championships Super Bowl XIV -> Candidate entity: ["Super Bowl XIV"]
[Path 5] Pittsburgh Steelers sports.sports_team.championships 2009 AFC Championship Game -> Candidate entity: ["2009 AFC Championship Game"]

Output: ["Super Bowl XLIII", "Super Bowl XL", "Super Bowl XIV"]

### Example 3 (Failure Case - Irrelevant Info)
Question: who is the governor of california 2010?
Candidate reasoning paths:
[Path 1] California location.location.contains Sacramento -> Candidate entity: ["Sacramento"]
[Path 2] California location.us_state.capital Sacramento -> Candidate entity: ["Sacramento"]

Output: ["None"]
---

### Current Task
Question: {Q}
Candidate reasoning paths:
{T}

Output:'''

SUBQUESTION_ANSWERING = '''
You are an expert of world knowledge with strong logical skills.
Given the following question and reasoning paths, select only the entities that can answer the question among the candidate entities. Output must be a Python-style list.
If no suitable answer is found among the candidate entities, return ["None"].

Question: {Q}
Candidate reasoning paths: 
{T}
Return :'''

#!#!
FINAL_ANSWER_PROMPT_new = '''
You are a logical reasoning system.
Your goal is to answer the Question based on the 'Candidate reasoning paths'.

CRITICAL PROTOCOL:
1. **Goal:** Maximize Recall. Include **ALL** entities that correctly answer the question.
2. **List Extraction:** If multiple paths look correct, output ALL of them. Do not stop at the first match.
3. **Filtering:** Distinguish between correct answers and clearly wrong types (e.g., "Super Bowl" vs. "AFC Championship"), but if a candidate is valid and just slightly ambiguous, INCLUDE it.
4. **Fallback:** If (and ONLY if) the candidate paths are empty, use your internal world knowledge.
5. **Format:** Output **ONLY** a valid Python-style list of strings.

---
### Example 1 (Filtering - Distinguish Exact Matches)
Question: what years have the steelers won the super bowl
Candidate reasoning paths:
Pittsburgh Steelers sports.sports_team.championships Super Bowl XLIII	Candidate entity : ["Super Bowl XLIII"]
Pittsburgh Steelers sports.sports_team.championships 2011 AFC Championship Game	Candidate entity : ["2011 AFC Championship Game"]
Pittsburgh Steelers sports.sports_team.championships Super Bowl XL	Candidate entity : ["Super Bowl XL"]
Pittsburgh Steelers sports.sports_team.championships Super Bowl XIV	Candidate entity : ["Super Bowl XIV"]

Output: ["Super Bowl XLIII", "Super Bowl XL", "Super Bowl XIV"]

### Example 2 (List Extraction - Countermeasure for Missing Answers)
Question: what years did the red sox win the world series
Candidate reasoning paths:
Boston Red Sox sports.sports_team.championships 2007 World Series	Candidate entity : ["2007 World Series"]
Boston Red Sox sports.sports_team.championships 1918 World Series	Candidate entity : ["1918 World Series"]
Boston Red Sox sports.sports_team.championships 2004 World Series	Candidate entity : ["2004 World Series"]
Boston Red Sox sports.sports_team.championships 1903 World Series	Candidate entity : ["1903 World Series"]

Output: ["2007 World Series", "1918 World Series", "2004 World Series", "1903 World Series"]

### Example 3 (Fallback to Internal Knowledge)
Question: when was bill clinton elected president
Candidate reasoning paths:

Output: ["1992", "1996"]
---

### Current Task
Question: {Q}
Candidate reasoning paths: 
{T}
Output:'''

FINAL_ANSWER_PROMPT = '''
You are an expert of world knowledge with strong logical skills.
Given the following question and reasoning paths, select only the entities that can answer the question among the candidate entities.
Output must be a Python-style list.

Question: {Q}
Candidate reasoning paths: 
{T}
Return :'''

GPT_NO_REASONING_PATH = '''
Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list (e.g., ["answer"]).
Question: {Q}
Return:'''

