# Architecture (Initial Skeleton)

This repository starts with a minimal full‑stack loop:
- **FastAPI** service exposing `/v1/score`
- **Streamlit** UI that calls the API
- **Risk scorer** using a simple heuristic (will be replaced by ML model + explanations)

Planned expansions:
- Data ingestion (World Bank / OCDS / GDELT), feature pipelines (Great Expectations)
- Graph features (NetworkX prototype → CosmosDB/Neo4j)
- ML training and registry (MLflow), monitoring
- Responsible AI docs (model cards, data sheets), provenance
