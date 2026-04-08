# LwM2M Implementation - Pre-Start Checklist

This document lists everything that must be decided, prepared, and validated before starting LwM2M code changes.

## Objective

Integrate outbound result reporting from the current YOLO + OCR pipeline to an LwM2M server with minimal risk to real-time inference performance.

## 1. Scope Decisions (Must Be Finalized First)

- Reporting mode:
  - Summary mode: send periodic aggregates when a number as been detected more than 5 times, we consider it viable and transmitable.
- Target script for first rollout:
  - Phase 1 target should be app_live_ocr.py only.
- Define acceptable latency for delivery:
  - Near-real-time (seconds) when network is available, store and forward if no network available.
- Define acceptable data loss policy when offline:
  - Store and forward if no network available.

## 2. Server and Security Prerequisites

- Confirm which LwM2M server will be used:
  - Local test server (example: Leshan) for validation.
  - Production endpoint for deployment.
- Confirm transport/security profile:
  - CoAP + DTLS with PSK (MVP).
  - Certificate-based authentication later.
- Confirm provisioning data is available:
  - Endpoint name format.
  - PSK identity.
  - PSK key rotation strategy.
  - Host and port policy.
- Confirm firewall/network path from Jetson container to server:
  - Outbound UDP for CoAP/DTLS path.

## 3. Data Contract Definition

- Finalize payload fields for each detection event:
  - device_id
  - source
  - timestamp_utc
  - frame_id
  - detected_number
  - rolling count for detected number
- Define serialization format:
  - JSON first (simple and debuggable), optional CBOR later.
- Define object/resource mapping convention:
  - Use custom object/resource IDs agreed with backend team.

## 4. Runtime and Reliability Requirements

- Define max sender queue size and memory budget.
- Define retry policy:
  - max retries
  - backoff schedule
  - timeout per send
- Define behavior during server outage:
  - continue local inference with non-blocking queue.
  - persist unsent events to disk.
- Define observability requirements:
  - sender success/failure counters
  - queue depth
  - reconnect count

## 5. Docker and Environment Readiness

- Add required Python dependency for LwM2M client implementation.
- Confirm dependency installation works in current container startup model.
- Define environment variable names in docker-compose:
  - LWM2M_SERVER_HOST
  - LWM2M_SERVER_PORT
  - LWM2M_ENDPOINT_NAME
  - LWM2M_PSK_ID
  - LWM2M_PSK_KEY
- Confirm secret handling approach:
  - no credentials in source files.

## 6. Code Integration Plan (Before Any Edit)

- Confirm insertion point for event publish in OCR loop.
- Confirm insertion point for periodic heartbeat in stats loop.
- Confirm shutdown flush behavior for graceful stop.
- Confirm that producer threads never wait on network I/O.
- Confirm that LwM2M sender can be disabled with config flag.

## 7. Test Plan Definition (Pre-Implementation)

- Define local functional tests:
  - registration success
  - first event publish
  - periodic summary publish
- Define failure-path tests:
  - server unavailable at startup
  - server loss during stream
  - queue overflow behavior
- Define regression checks for existing behavior:
  - current OCR counting unchanged
  - CSV output unchanged
  - frame processing rate not significantly degraded

## 8. Acceptance Criteria (Go/No-Go)

Implementation can start only when all below are confirmed:

- Scope and reporting mode approved.
- Security mode and credentials flow approved.
- Payload contract approved by backend consumer.
- Retry and buffering policies approved.
- Test server reachable from container runtime.
- Rollback strategy documented (LwM2M disabled path).

## 9. Phase 1 Implementation Status

Current status: started.

- Added a dedicated Phase 1 reporter module with:
  - non-blocking in-memory queue
  - store-and-forward persistence to local disk
  - CoAP summary publish path
- Integrated reporter wiring in app_live_ocr.py behind explicit flags.
- Added runtime options for server URI, endpoint, device id, threshold, interval, and store file.
- Added aiocoap dependency and runner script dependency check/update.

Next implementation steps:

- Validate against a local Leshan endpoint.
- Confirm object/resource URI expected by backend.
- Add DTLS PSK mode for secure transport.
- Add basic integration test scenario for offline store-and-forward replay.
