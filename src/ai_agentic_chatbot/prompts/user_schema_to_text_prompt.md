Transform the following table schema into JSON that STRICTLY conforms
to the following Pydantic model:

TableSchemaDocumentation:
- table_name: string
- business_purpose: string
- primary_identifier: string
- key_fields: list of ( field_name, meaning )
- important_dates: list of ( field_name, meaning ) OR null
- relationships: list of ( related_table, explanation ) OR null
- operational_notes: string OR null
- example_questions: list of strings

Rules:
- Output ONLY valid JSON
- Do NOT include markdown
- Do NOT include comments
- Do NOT include explanations outside JSON

Table schema:
{table_json}