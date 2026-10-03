"""Public TIME replay-to-window and alert integration."""

import asyncio
from datetime import timedelta

from geo_infer_time import ReplayIngestAdapter, StreamProcessor


class TestTimeIntegration:
    def test_replay_updates_windows_watermark_and_alert_handlers(self) -> None:
        processor = StreamProcessor(
            window_size=timedelta(minutes=1), watermark_delay=timedelta(seconds=2)
        )
        records = [
            {"timestamp": second, "value": 100 if second == 10 else 1}
            for second in range(20)
        ]
        alerts = []
        processor.register_anomaly_alert_handler(alerts.append)
        assert (
            asyncio.run(
                processor.ingest_adapter_stream(
                    ReplayIngestAdapter(records, {"timestamp_unit": "s"})
                )
            )
            == 20
        )
        processor.process_sliding_window_anomaly_alerts(z_threshold=3)
        assert len(alerts) == 1
        assert alerts[0]["value"] == 100
        assert processor.get_watermark().timestamp() == 17
        assert processor.process_session_windows(timedelta(seconds=1))[0]["count"] == 20
        assert processor.process_window()["aggregated_value"] == 5.95
