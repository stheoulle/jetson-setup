#!/usr/bin/env python3
"""Smoke test for Phase 1 LwM2M summary reporter."""

import time

from lwm2m_phase1 import LwM2MSummaryReporter


def main():
    reporter = LwM2MSummaryReporter(
        enabled=True,
        server_uri="coap://127.0.0.1:5683/lwm2m/summary",
        endpoint_name="jetson-ocr-test",
        device_id="jetson-ocr-test-device",
        source="smoketest",
        threshold=5,
        interval_sec=2,
        store_file="lwm2m_pending_test.ndjson",
    )

    if not reporter.is_active():
        print("Reporter inactive. Ensure aiocoap is installed.")
        return 1

    reporter.start()

    counts = {"0379": 3, "1008": 5}
    payload = reporter.build_summary_payload(counts, frame_count=50, processed_count=45, detection_count=8)
    print("First payload expected: only number 1008")
    reporter.enqueue(payload)

    time.sleep(3)

    counts = {"0379": 6, "1008": 7, "2001": 1}
    payload = reporter.build_summary_payload(counts, frame_count=120, processed_count=100, detection_count=14)
    print("Second payload expected: number 0379 and delta for 1008")
    reporter.enqueue(payload)

    time.sleep(4)

    reporter.stop()
    stats = reporter.get_stats()
    print("[SMOKETEST] sender stats:", stats)
    print("[SMOKETEST] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
