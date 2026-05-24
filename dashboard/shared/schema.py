"""Single source of truth for cross-language schemas.

Re-exports the Pydantic models defined in the backend so that
`openapi-typescript` can crawl `/openapi.json` and generate
`shared/schema.ts`. See `scripts/build.sh`.
"""
from backend.app import (  # noqa: F401
    DiagnosticsResponse,
    ManifoldPoint,
    ManifoldResponse,
    SteerRequest,
    SteerResponse,
)
