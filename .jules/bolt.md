## 2024-05-22 - Optimizing List Queries
**Learning:** Fetching large text fields (like `call_context`) in list endpoints significantly impacts performance (10x slower). Always select specific columns for list views.
**Action:** When creating or modifying list endpoints, always review the schema for large fields and exclude them from the initial query unless absolutely necessary.
