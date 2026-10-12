from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from multi_stream_asr.config import AppConfig
from multi_stream_asr.log.recorder import RunRecorder
from multi_stream_asr.models import EventStatus
from multi_stream_asr.pipeline.process_manager import ProcessPipelineManager
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.web.event_bus import FrontendEventBus


class HeadlessRunner:
    def __init__(
        self,
        *,
        config: AppConfig,
        fake: bool = False,
        event_bus: FrontendEventBus | None = None,
        recorder: RunRecorder | None = None,
    ) -> None:
        self._config = config
        self._fake = fake
        self._event_bus = event_bus or FrontendEventBus()
        self._recorder = recorder or RunRecorder()
        self.manager: Any | None = None
        self.last_run_dir: Path | None = None

    async def run(self) -> int:
        self.manager = self.build_manager()
        models_loaded = False
        try:
            await self.manager.load_models()
            models_loaded = True
            await self.manager.start_recognition()
            await self.manager.wait_for_completion()
            self.last_run_dir = self._save_results()
            await self.manager.unload_models()
        except Exception:
            if self.manager is not None and models_loaded:
                self.last_run_dir = self._save_results()
            if self.manager is not None and self.manager.state is not PipelineState.UNLOADED:
                try:
                    await self.manager.unload_models()
                except Exception:
                    pass
            return 1
        return 0

    def build_manager(self) -> Any:
        return self._build_manager()

    def _build_manager(self) -> ProcessPipelineManager:
        return ProcessPipelineManager(
            config=self._config,
            event_bus=self._event_bus,
            fake=self._fake,
        )

    def _save_results(self) -> Path:
        assert self.manager is not None
        run_dir = getattr(self.manager, "run_dir", None)
        if run_dir is None:
            run_dir = self._config.output_dir / f"headless-{int(time.time() * 1000)}"
        lane_summaries = getattr(self.manager, "lane_summaries", None)
        if lane_summaries is not None:
            return self._recorder.save_process_run(
                run_dir,
                config=self._config,
                lane_summaries=lane_summaries,
            )
        return self._recorder.save(
            run_dir,
            config=self._config,
            lane_metrics=self.manager.lane_metrics,
            latency_tracker=self.manager.latency_tracker,
            rtf_tracker=self.manager.rtf_tracker,
            event_loop_lag_tracker=self.manager.event_loop_lag_tracker,
            results={
                key: value
                for key, value in self._event_bus.snapshot_results().items()
                if value["status"] is not EventStatus.LANE_FINISHED
            },
        )
