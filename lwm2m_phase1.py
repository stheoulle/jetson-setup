#!/usr/bin/env python3
"""Phase 1 LwM2M summary reporter with store-and-forward behavior."""

import asyncio
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import aiocoap
except Exception:
    aiocoap = None


class LwM2MSummaryReporter:
    """Non-blocking summary sender that persists unsent payloads to disk."""

    def __init__(
        self,
        enabled,
        server_uri,
        endpoint_name,
        device_id,
        source,
        threshold=5,
        interval_sec=5,
        queue_size=200,
        retry_delay_sec=5,
        store_file="lwm2m_pending.ndjson",
    ):
        self.enabled = bool(enabled)
        self.server_uri = server_uri
        self.endpoint_name = endpoint_name
        self.device_id = device_id
        self.source = source
        self.threshold = max(1, int(threshold))
        self.interval_sec = max(1, int(interval_sec))
        self.retry_delay_sec = max(1, int(retry_delay_sec))
        self.out_queue = queue.Queue(maxsize=max(10, int(queue_size)))

        self.store_file = Path(store_file)
        self.store_file.parent.mkdir(parents=True, exist_ok=True)

        self.stop_event = threading.Event()
        self.worker = None
        self.pending = deque(self._load_pending())
        self.last_reported_counts = {}

        self.sent_count = 0
        self.failed_count = 0
        self.queued_count = 0

    def is_active(self):
        return self.enabled and aiocoap is not None and bool(self.server_uri)

    def start(self):
        if not self.is_active():
            return
        if self.worker and self.worker.is_alive():
            return
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=3)

    def build_summary_payload(self, counts, frame_count, processed_count, detection_count):
        eligible = []
        for number, count in sorted(counts.items()):
            if count < self.threshold:
                continue
            previous = self.last_reported_counts.get(number, 0)
            if count <= previous:
                continue
            eligible.append(
                {
                    "number": number,
                    "count": int(count),
                    "delta": int(count - previous),
                }
            )

        if not eligible:
            return None

        for item in eligible:
            self.last_reported_counts[item["number"]] = item["count"]

        return {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "device_id": self.device_id,
            "endpoint_name": self.endpoint_name,
            "source": self.source,
            "mode": "summary",
            "threshold": self.threshold,
            "stats": {
                "frame_count": int(frame_count),
                "processed_count": int(processed_count),
                "detection_count": int(detection_count),
                "unique_numbers": int(len(counts)),
            },
            "numbers": eligible,
        }

    def enqueue(self, payload):
        if not self.is_active() or payload is None:
            return False

        while True:
            try:
                self.out_queue.put_nowait(payload)
                self.queued_count += 1
                return True
            except queue.Full:
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    return False

    def get_stats(self):
        return {
            "sent": self.sent_count,
            "failed": self.failed_count,
            "queued": self.queued_count,
            "pending_disk": len(self.pending),
            "queue_depth": self.out_queue.qsize(),
        }

    def _load_pending(self):
        if not self.store_file.exists():
            return []

        items = []
        try:
            with self.store_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []

        return items

    def _persist_pending(self):
        tmp_path = self.store_file.with_suffix(self.store_file.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for item in self.pending:
                    f.write(json.dumps(item, separators=(",", ":")) + "\n")
            tmp_path.replace(self.store_file)
        except Exception:
            pass

    async def _send_once(self, payload):
        request_payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        context = await aiocoap.Context.create_client_context()
        try:
            req = aiocoap.Message(code=aiocoap.POST, uri=self.server_uri, payload=request_payload)
            response = await asyncio.wait_for(context.request(req).response, timeout=8)
            return response.code.is_successful()
        finally:
            await context.shutdown()

    def _run_worker(self):
        if self.pending:
            self._persist_pending()

        while not self.stop_event.is_set():
            payload = None
            payload_from_pending = False

            if self.pending:
                payload = self.pending.popleft()
                payload_from_pending = True
            else:
                try:
                    payload = self.out_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

            try:
                ok = asyncio.run(self._send_once(payload))
            except Exception:
                ok = False

            if ok:
                self.sent_count += 1
                if payload_from_pending:
                    self._persist_pending()
                continue

            self.failed_count += 1
            self.pending.append(payload)
            self._persist_pending()
            time.sleep(self.retry_delay_sec)
