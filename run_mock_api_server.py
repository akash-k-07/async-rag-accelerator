"""
Entry point for running the mock API server.
"""

import logging
from spec_rag.tools.mock_api_server import get_shipment_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MockAPI] %(message)s"
)

if __name__ == "__main__":
    logging.info("Initializing Mock API Server (simulated)...")
    # In a real scenario, you'd start a Flask/FastAPI app here.
    # For now, we just demonstrate it's callable.
    print(get_shipment_status("TRK1234"))
