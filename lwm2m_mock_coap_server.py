#!/usr/bin/env python3
"""Minimal CoAP server to receive Phase 1 summary payloads."""

import asyncio
import logging

import aiocoap.resource as resource
import aiocoap


class SummaryResource(resource.Resource):
    async def render_post(self, request):
        payload = request.payload.decode("utf-8", errors="replace")
        print("\n[MOCK SERVER] POST /lwm2m/summary")
        print(payload)
        return aiocoap.Message(code=aiocoap.CHANGED, payload=b"ok")


async def main():
    root = resource.Site()
    root.add_resource(["lwm2m", "summary"], SummaryResource())

    await aiocoap.Context.create_server_context(root, bind=("0.0.0.0", 5683))
    print("[MOCK SERVER] Listening on coap://0.0.0.0:5683/lwm2m/summary")

    await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
