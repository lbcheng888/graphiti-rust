//! Prompt templates for various graph operations

/// Template for extracting entities from conversational messages
pub const EXTRACT_NODES_MESSAGE_TEMPLATE: &str = r#"
You are an AI assistant that extracts entity nodes from conversational messages.
Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation.

<ENTITY TYPES>
{{entity_types}}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{{#if previous_episodes}}
{{#each previous_episodes}}
{{this}}
{{/each}}
{{else}}
[]
{{/if}}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{{episode_content}}
</CURRENT MESSAGE>

Instructions:
1. Extract entities that are explicitly or implicitly mentioned in the CURRENT MESSAGE
2. For each entity extracted, determine its entity type based on the provided ENTITY TYPES
3. Indicate the classified entity type by providing its entity_type_id
4. You may use information from PREVIOUS MESSAGES only to disambiguate references or support continuity
5. Do not extract entities that are not mentioned in the CURRENT MESSAGE

{{#if custom_prompt}}
{{custom_prompt}}
{{/if}}

Return a JSON object with the following structure:
{
  "extracted_entities": [
    {
      "name": "entity name",
      "entity_type_id": 1,
      "confidence": 0.95
    }
  ]
}
"#;

/// Template for extracting entities from JSON data
pub const EXTRACT_NODES_JSON_TEMPLATE: &str = r#"
You are an AI assistant that extracts entity nodes from JSON.
Your primary task is to extract and classify relevant entities from JSON files.

<ENTITY TYPES>
{{entity_types}}
</ENTITY TYPES>

<SOURCE DESCRIPTION>
{{source_description}}
</SOURCE DESCRIPTION>

<JSON>
{{episode_content}}
</JSON>

{{#if custom_prompt}}
{{custom_prompt}}
{{/if}}

Given the above source description and JSON, extract relevant entities from the provided JSON.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

Return a JSON object with the following structure:
{
  "extracted_entities": [
    {
      "name": "entity name",
      "entity_type_id": 1,
      "confidence": 0.95
    }
  ]
}
"#;

/// Template for extracting entities from general text
pub const EXTRACT_NODES_TEXT_TEMPLATE: &str = r#"
You are an AI assistant that extracts entity nodes from text.
Your primary task is to extract and classify the speaker and other significant entities mentioned in the provided text.

<ENTITY TYPES>
{{entity_types}}
</ENTITY TYPES>

<TEXT>
{{episode_content}}
</TEXT>

Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

{{#if custom_prompt}}
{{custom_prompt}}
{{/if}}

Return a JSON object with the following structure:
{
  "extracted_entities": [
    {
      "name": "entity name",
      "entity_type_id": 1,
      "confidence": 0.95
    }
  ]
}
"#;

/// Template for node classification
pub const CLASSIFY_NODES_TEMPLATE: &str = r#"
You are an AI assistant that classifies entity nodes based on their content and context.

<ENTITY TYPES>
{{entity_types}}
</ENTITY TYPES>

<NODE TO CLASSIFY>
{{node_data}}
</NODE TO CLASSIFY>

<CONTEXT>
{{context}}
</CONTEXT>

Given the above node data and context, classify the entity into one of the provided entity types.
Consider the node's attributes, relationships, and contextual information.

Return a JSON object with the following structure:
{
  "entity_type_id": 1,
  "confidence": 0.95,
  "reasoning": "Explanation for the classification"
}
"#;

/// Template for extracting node attributes
pub const EXTRACT_ATTRIBUTES_TEMPLATE: &str = r#"
You are an AI assistant that extracts structured attributes from entity descriptions.

<ENTITY>
{{entity_data}}
</ENTITY>

<CONTEXT>
{{context}}
</CONTEXT>

<ATTRIBUTE SCHEMA>
{{attribute_schema}}
</ATTRIBUTE SCHEMA>

Given the above entity and context, extract structured attributes according to the provided schema.
Only extract attributes that are explicitly mentioned or can be reliably inferred.

Return a JSON object with the extracted attributes:
{
  "attributes": {
    "attribute_name": "value",
    "another_attribute": "another_value"
  }
}
"#;

/// Template for extracting relationships/edges from text
pub const EXTRACT_EDGES_TEMPLATE: &str = r#"
You are an expert fact extractor that extracts fact triples from text.
1. Extracted fact triples should also be extracted with relevant date information.
2. Treat the CURRENT TIME as the time the CURRENT MESSAGE was sent. All temporal information should be extracted relative to this time.

<FACT TYPES>
{{edge_types}}
</FACT TYPES>

<PREVIOUS_MESSAGES>
{{#if previous_episodes}}
{{#each previous_episodes}}
{{this}}
{{/each}}
{{else}}
[]
{{/if}}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{{episode_content}}
</CURRENT_MESSAGE>

<ENTITIES>
{{nodes}}
</ENTITIES>

<REFERENCE_TIME>
{{reference_time}}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>

Extract fact triples from the CURRENT_MESSAGE that involve entities from ENTITIES.
You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.

{{#if custom_prompt}}
{{custom_prompt}}
{{/if}}

# EXTRACTION RULES

1. Only emit facts where both the subject and object match IDs in ENTITIES.
2. Each fact must involve two **distinct** entities.
3. Use a SCREAMING_SNAKE_CASE string as the `relation_type` (e.g., FOUNDED, WORKS_AT).
4. Do not emit duplicate or semantically redundant facts.
5. The `fact_text` should quote or closely paraphrase the original source sentence(s).
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week").
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# DATETIME RULES

- `valid_from`: When the fact became true (ISO 8601 UTC).
- `valid_to`: When the fact stopped being true (ISO 8601 UTC). Use `null` if ongoing.
- For ongoing facts, set `valid_to` to `null`.
- For past facts, provide both `valid_from` and `valid_to` if determinable.

Return a JSON object with the following structure:
{
  "extracted_edges": [
    {
      "source_entity_id": "uuid",
      "target_entity_id": "uuid",
      "relation_type": "SCREAMING_SNAKE_CASE",
      "fact_text": "Direct quote or close paraphrase",
      "valid_from": "2024-01-01T00:00:00Z",
      "valid_to": null,
      "confidence": 0.9
    }
  ]
}
"#;

/// Template for edge reflexion (validation and correction)
pub const EXTRACT_EDGES_REFLEXION_TEMPLATE: &str = r#"
You are an expert fact validator. Your task is to validate and potentially correct extracted facts.

<ORIGINAL MESSAGE>
{{episode_content}}
</ORIGINAL MESSAGE>

<EXTRACTED FACTS>
{{extracted_facts}}
</EXTRACTED FACTS>

<REFERENCE TIME>
{{reference_time}}
</REFERENCE TIME>

Review the extracted facts and check for:
1. Accuracy against the original message
2. Proper temporal information
3. Correct entity references
4. Appropriate confidence scores

Return a JSON object with corrected facts:
{
  "validated_edges": [
    {
      "source_entity_id": "uuid",
      "target_entity_id": "uuid",
      "relation_type": "SCREAMING_SNAKE_CASE",
      "fact_text": "Corrected fact text",
      "valid_from": "2024-01-01T00:00:00Z",
      "valid_to": null,
      "confidence": 0.9
    }
  ]
}
"#;

/// Template for extracting edge attributes
pub const EXTRACT_EDGE_ATTRIBUTES_TEMPLATE: &str = r#"
You are an expert at extracting detailed attributes from relationships.

<MESSAGE>
{{episode_content}}
</MESSAGE>

<REFERENCE TIME>
{{reference_time}}
</REFERENCE TIME>

Given the above MESSAGE, its REFERENCE TIME, and the following FACT, update any of its attributes based on the information provided in MESSAGE. Use the provided attribute descriptions to better understand how each attribute should be determined.

Guidelines:
1. Do not hallucinate entity property values if they cannot be found in the current context.
2. Only use the provided MESSAGES and FACT to set attribute values.

<FACT>
{{fact}}
</FACT>

Return a JSON object with updated fact attributes:
{
  "updated_fact": {
    "source_entity_id": "uuid",
    "target_entity_id": "uuid",
    "relation_type": "SCREAMING_SNAKE_CASE",
    "attributes": {
      "key": "value"
    },
    "confidence": 0.9
  }
}
"#;

/// Template for single node deduplication
pub const DEDUPE_NODES_TEMPLATE: &str = r#"
You are a helpful assistant that determines whether or not a NEW ENTITY is a duplicate of any EXISTING ENTITIES.

<PREVIOUS MESSAGES>
{{#if previous_episodes}}
{{#each previous_episodes}}
{{this}}
{{/each}}
{{else}}
[]
{{/if}}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{{episode_content}}
</CURRENT MESSAGE>

<NEW ENTITY>
{{extracted_node}}
</NEW ENTITY>

<ENTITY TYPE DESCRIPTION>
{{entity_type_description}}
</ENTITY TYPE DESCRIPTION>

<EXISTING ENTITIES>
{{existing_nodes}}
</EXISTING ENTITIES>

Given the above EXISTING ENTITIES and their attributes, MESSAGE, and PREVIOUS MESSAGES; Determine if the NEW ENTITY extracted from the conversation is a duplicate entity of one of the EXISTING ENTITIES.

Guidelines:
1. Two entities are duplicates if they refer to the same real-world entity
2. Consider variations in naming, spelling, and representation
3. Use context from messages to disambiguate
4. Be conservative - only mark as duplicate if you are confident

Return a JSON object with the following structure:
{
  "is_duplicate": true,
  "duplicate_of_idx": 2,
  "confidence": 0.95,
  "reasoning": "Explanation for the decision"
}

If the entity is NOT a duplicate, set "is_duplicate" to false and "duplicate_of_idx" to -1.
"#;

/// Template for node list deduplication
pub const DEDUPE_NODES_LIST_TEMPLATE: &str = r#"
You are a helpful assistant that de-duplicates nodes from node lists.

Given the following context, deduplicate a list of nodes:

Nodes:
{{nodes}}

Task:
1. Group nodes together such that all duplicate nodes are in the same list of uuids
2. All duplicate uuids should be grouped together in the same list
3. Also return a new summary that synthesizes the summary into a new short summary

Guidelines:
1. Each uuid from the list of nodes should appear EXACTLY once in your response
2. If a node has no duplicates, it should appear in the response in a list of only one uuid

Respond with a JSON object in the following format:
{
  "nodes": [
    {
      "uuids": ["5d643020624c42fa9de13f97b1b3fa39", "node that is a duplicate of 5d643020624c42fa9de13f97b1b3fa39"],
      "summary": "Brief summary of the node summaries that appear in the list of names."
    }
  ]
}
"#;

/// Template for bulk node deduplication
pub const DEDUPE_NODES_BULK_TEMPLATE: &str = r#"
You are a helpful assistant that de-duplicates nodes from node lists.

<NODES TO DEDUPLICATE>
{{nodes}}
</NODES TO DEDUPLICATE>

<CONTEXT>
{{context}}
</CONTEXT>

Task: Group nodes that refer to the same real-world entity.

Guidelines:
1. Carefully analyze each node's name, type, and attributes
2. Group nodes that clearly refer to the same entity
3. Consider variations in naming, spelling, and representation
4. Be conservative - only group if you are confident they are the same
5. Each node UUID should appear exactly once in the response

Return a JSON object with grouped nodes:
{
  "duplicate_groups": [
    {
      "uuids": ["uuid1", "uuid2"],
      "merged_summary": "Combined summary for the group",
      "confidence": 0.95
    }
  ]
}
"#;

/// Template for deduplicating edges
pub const DEDUPE_EDGES_TEMPLATE: &str = r#"
You are an expert at identifying duplicate relationships. Your task is to find groups of edges that represent the same relationship.

## Edges to Analyze:
{{#each items}}
Edge {{@index}}: {{this}}
{{/each}}

## Context:
{{context}}

## Instructions:
1. Carefully analyze each edge/relationship
2. Group edges that represent the same relationship between entities
3. Consider semantic equivalence, not just exact matches
4. Only group edges if you are confident they are duplicates
5. Provide clear explanation for your grouping decisions

## Response Format:
Return a JSON object with the following structure:
{
  "duplicate_groups": [
    ["edge_uuid_1", "edge_uuid_2"],
    ["edge_uuid_3", "edge_uuid_4"]
  ],
  "explanation": "Detailed explanation of grouping logic"
}

Identify duplicate edge groups now:
"#;

/// Template for summarizing nodes
pub const SUMMARIZE_NODES_TEMPLATE: &str = r#"
You are an expert at creating concise, informative summaries. Your task is to update an entity's summary with new information.

## Current Node Data:
{{node_data}}

## New Information to Incorporate:
{{new_information}}

{{#if previous_summary}}
## Previous Summary:
{{previous_summary}}
{{/if}}

## Instructions:
1. Analyze the current node data and new information
2. Create or update a comprehensive summary that captures key attributes
3. Extract important attributes as structured data
4. Keep the summary concise but informative
5. Maintain consistency with previous information unless contradicted

## Response Format:
Return a JSON object with the following structure:
{
  "summary": "Updated comprehensive summary of the entity",
  "attributes": {
    "key1": "value1",
    "key2": "value2"
  }
}

Create the updated summary now:
"#;

/// Template for invalidating edges
pub const INVALIDATE_EDGES_TEMPLATE: &str = r#"
You are an expert at identifying when relationships become invalid due to new information. Your task is to determine which existing edges should be invalidated based on new edges.

## New Edges:
{{#each new_edges}}
New Edge {{@index}}: {{this}}
{{/each}}

## Existing Edges to Check:
{{#each existing_edges}}
Existing Edge {{@index}}: {{this}}
{{/each}}

## Episode Context:
{{episode_context}}

## Instructions:
1. Analyze the new edges and their implications
2. Identify existing edges that are contradicted or superseded by new information
3. Consider temporal aspects - newer information may invalidate older relationships
4. Only invalidate edges if there is clear contradiction or supersession
5. Provide clear explanation for invalidation decisions

## Response Format:
Return a JSON object with the following structure:
{
  "invalidated_edges": ["edge_uuid_1", "edge_uuid_2"],
  "explanation": "Detailed explanation of why these edges were invalidated"
}

Identify edges to invalidate now:
"#;
