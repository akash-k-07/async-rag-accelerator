"""Generic mock API server tool definitions."""

import time
import logging
import json
import uuid
import datetime


def get_shipment_status(tracking_id: str) -> str:
    """
    Simulates a high-latency enterprise API endpoint for shipment tracking.
    Injects a deterministic 3000ms latency to simulate legacy constraints.
    """
    logging.info(
        "Received API request for tracking_id: %s. "
        "Inducing structural latency...",
        tracking_id
    )

    # Simulated I/O bottleneck
    time.sleep(3.0)

    logging.info(
        "Latency complete. Returning ground truth for %s.", tracking_id
    )

    return json.dumps({
        "tracking_id": tracking_id,
        "status": "out_for_delivery",
        "estimated_arrival": "17:00",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "trace_id": str(uuid.uuid4())
    })
