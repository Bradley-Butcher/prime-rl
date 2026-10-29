"""The orchestrator's curriculum plan routes, served by its control API."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from prime_rl.orchestrator.curriculum.plans import (
    PlanConflictError,
    PlanError,
    PlanKey,
    PlanNotFoundError,
    PlanRequest,
    Plans,
)
from prime_rl.orchestrator.curriculum.shaping import Shaping, ShapingRequest


def plan_router(
    plans: Plans, run_status: Callable[[], dict[str, Any]] = dict, shaping: Shaping | None = None
) -> APIRouter:
    """Routes to post, read and delete plans, and to read the run's status and stats. ``run_status`` adds the
    orchestrator's own state to ``/v1/status``. With ``shaping``, also the routes to change and read the reward
    shaping, whose setting ``/v1/status`` then shows."""
    router = APIRouter()

    @router.post("/v1/plans")
    async def set_plan(request: PlanRequest) -> Any:
        return _respond(lambda: plans.set(request.step, request.plan, author=request.author, note=request.note))

    @router.get("/v1/plans")
    async def list_plans() -> list[dict[str, Any]]:
        return plans.all()

    @router.get("/v1/plans/{step}")
    async def get_plan(step: str) -> Any:
        return _respond(lambda: plans.get(_plan_key(step)))

    @router.delete("/v1/plans/{step}")
    async def delete_plan(step: str, author: str | None = None, note: str | None = None) -> Any:
        return _respond(lambda: plans.delete(_plan_key(step), author=author, note=note))

    @router.get("/v1/status")
    async def get_status() -> dict[str, Any]:
        return plans.status() | ({"shaping": shaping.status()} if shaping is not None else {}) | run_status()

    if shaping is not None:

        @router.post("/v1/shaping")
        async def set_shaping(request: ShapingRequest) -> Any:
            return _respond(lambda: shaping.set(request))

        @router.get("/v1/shaping")
        async def get_shaping() -> dict[str, Any]:
            return shaping.describe()

    @router.get("/v1/stats")
    async def get_stats(
        step: int | None = Query(None, ge=1),
        env: str | None = None,
        label: list[str] | None = Query(None),
        limit: int = Query(100, ge=0),
        top_labels: int = Query(20, ge=0),
    ) -> dict[str, Any]:
        return plans.stats(step=step, env=env, labels=label, limit=limit, top_labels=top_labels)

    return router


def _respond(call: Callable[[], dict[str, Any]]) -> dict[str, Any] | JSONResponse:
    """The call's result, or its plan error as a JSON error body. A conflict names the step dispatching and
    the first step still open, for a client to post for that instead."""
    try:
        return call()
    except PlanConflictError as error:
        content = {
            "detail": str(error),
            "current_dispatch_step": error.current_dispatch_step,
            "next_open_step": error.next_open_step,
        }
        return JSONResponse(content, status_code=409)
    except PlanNotFoundError as error:
        return JSONResponse({"detail": str(error)}, status_code=404)
    except PlanError as error:
        return JSONResponse({"detail": str(error)}, status_code=422)


def _plan_key(step: str) -> PlanKey:
    if step == "default":
        return step
    if step.isdigit() and int(step) >= 1:
        return int(step)
    raise PlanError(f"A plan's step is a step number or 'default', got {step!r}")
