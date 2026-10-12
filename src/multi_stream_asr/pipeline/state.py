from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    UNLOADING = "unloading"
    FATAL = "fatal"


_LEGAL_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.UNLOADED: {PipelineState.LOADING},
    PipelineState.LOADING: {PipelineState.LOADED, PipelineState.FATAL},
    PipelineState.LOADED: {PipelineState.STARTING, PipelineState.UNLOADING, PipelineState.FATAL},
    PipelineState.STARTING: {PipelineState.RUNNING, PipelineState.STOPPING, PipelineState.FATAL},
    PipelineState.RUNNING: {PipelineState.STOPPING, PipelineState.FATAL},
    PipelineState.STOPPING: {
        PipelineState.LOADED,
        PipelineState.STOPPED,
        PipelineState.FATAL,
    },
    PipelineState.STOPPED: {PipelineState.STARTING, PipelineState.UNLOADING, PipelineState.FATAL},
    PipelineState.UNLOADING: {PipelineState.UNLOADED, PipelineState.FATAL},
    PipelineState.FATAL: {PipelineState.UNLOADED},
}


def assert_transition(current: PipelineState, target: PipelineState) -> None:
    legal_targets = _LEGAL_TRANSITIONS.get(current, set())
    if target not in legal_targets:
        raise ValueError(f"illegal pipeline state transition: {current.value} -> {target.value}")
