"""
MeshProtocolSimulator
=====================
Rigorous BLE/Wi-Fi Direct mesh network simulator for offline emergency
data relay through passing vehicles.

Responsibilities:
  - Binary packet encoding (struct.pack) with CRC-32 checksum.
  - Ad-hoc hop topology simulation (3–7 relay nodes).
  - Packet loss (15% per hop), bit-flip corruption (5% per hop).
  - Automatic checksum validation and retransmission logic (max 3 retries).
  - Produces a structured transmission log for display.
"""

from __future__ import annotations

import asyncio
import binascii
import logging
import random
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACKET_LOSS_PROBABILITY   = 0.15   # 15% per hop
BIT_FLIP_PROBABILITY      = 0.05   # 5% per hop
MAX_RETRANSMISSIONS       = 3
HOP_LATENCY_MS_MIN        = 20
HOP_LATENCY_MS_MAX        = 120
RADIO_RANGE_METRES        = 50

# Packet format (binary struct layout):
#   lat        : float  (4 bytes)  — latitude
#   lon        : float  (4 bytes)  — longitude
#   urgency    : float  (4 bytes)  — urgency index
#   timestamp  : uint32 (4 bytes)  — epoch seconds
#   sequence   : uint16 (2 bytes)  — sequence number
#   flags      : uint8  (1 byte)   — bit flags
#   reserved   : uint8  (1 byte)   — padding
# Total payload: 20 bytes + 4 bytes CRC-32 = 24 bytes
STRUCT_FORMAT = "!fffIHBB"   # network byte order
STRUCT_SIZE   = struct.calcsize(STRUCT_FORMAT)  # 20 bytes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MeshPacket:
    latitude:   float
    longitude:  float
    urgency:    float
    timestamp:  int
    sequence:   int   = 0
    flags:      int   = 0b00000001  # bit0 = emergency flag


@dataclass
class HopLog:
    hop_number:        int
    relay_node_id:     str
    distance_to_next:  float       # metres (simulated)
    latency_ms:        float
    status:            str         # "PASS" | "CORRUPTED" | "LOST" | "RETRANSMIT_OK" | "RETRANSMIT_FAIL"
    retransmit_count:  int
    rssi_dbm:          float       # simulated signal strength
    crc_expected:      str
    crc_received:      str
    notes:             str


@dataclass
class TransmissionResult:
    success:         bool
    total_hops:      int
    hops_completed:  int
    hop_logs:        list[HopLog]
    delivery_ratio:  float        # 0.0 – 1.0
    total_latency_ms: float
    raw_payload_hex: str
    final_status:    str          # "DELIVERED" | "FAILED"


# ---------------------------------------------------------------------------
# Encoding & checksum
# ---------------------------------------------------------------------------

def _encode_packet(pkt: MeshPacket) -> bytes:
    """Serialize packet to binary and append CRC-32."""
    payload = struct.pack(
        STRUCT_FORMAT,
        pkt.latitude,
        pkt.longitude,
        pkt.urgency,
        pkt.timestamp,
        pkt.sequence & 0xFFFF,
        pkt.flags & 0xFF,
        0,   # reserved
    )
    crc = struct.pack("!I", binascii.crc32(payload) & 0xFFFFFFFF)
    return payload + crc


def _decode_and_verify(raw: bytes) -> tuple[bool, str, str]:
    """
    Verify CRC-32. Returns (is_valid, expected_crc_hex, received_crc_hex).
    """
    if len(raw) < STRUCT_SIZE + 4:
        return False, "N/A", "TRUNCATED"

    payload  = raw[:STRUCT_SIZE]
    crc_recv = raw[STRUCT_SIZE:]

    expected_crc = binascii.crc32(payload) & 0xFFFFFFFF
    received_crc = struct.unpack("!I", crc_recv)[0]

    return (
        expected_crc == received_crc,
        f"{expected_crc:08X}",
        f"{received_crc:08X}",
    )


def _flip_random_bit(data: bytes) -> bytes:
    """Flip a single random bit in the byte string to simulate RF corruption."""
    data_arr = bytearray(data)
    byte_idx = random.randint(0, len(data_arr) - 1)
    bit_idx  = random.randint(0, 7)
    data_arr[byte_idx] ^= (1 << bit_idx)
    return bytes(data_arr)


# ---------------------------------------------------------------------------
# Topology generator
# ---------------------------------------------------------------------------

def _generate_hop_topology(num_hops: int) -> list[str]:
    """
    Generate relay node IDs simulating passing vehicles in radio range.
    """
    prefixes = ["VAN", "CAR", "BUS", "TRK", "MCY", "SUV"]
    return [
        f"{random.choice(prefixes)}-{random.randint(100, 999)}"
        for _ in range(num_hops)
    ]


def _simulate_rssi() -> float:
    """Simulated RSSI in dBm. Range: -40 (excellent) to -85 (poor)."""
    return round(random.uniform(-85.0, -40.0), 1)


def _simulate_distance() -> float:
    """Distance to next relay node in metres (within radio range)."""
    return round(random.uniform(5.0, RADIO_RANGE_METRES), 1)


# ---------------------------------------------------------------------------
# Core simulator (async for I/O latency simulation)
# ---------------------------------------------------------------------------

class MeshProtocolSimulator:
    """
    Simulates BLE/Wi-Fi Direct offline mesh relay of an emergency packet
    through ad-hoc vehicle relay nodes.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        self._sequence_counter = 0

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def transmit(
        self,
        latitude:  float,
        longitude: float,
        urgency:   float,
        num_hops:  Optional[int] = None,
    ) -> TransmissionResult:
        """
        Asynchronously simulate end-to-end mesh packet transmission.

        Parameters
        ----------
        latitude, longitude : float — incident coordinates
        urgency             : float — urgency index 1–10
        num_hops            : int   — relay nodes (default: random 3–7)
        """
        self._sequence_counter += 1
        pkt = MeshPacket(
            latitude=round(latitude, 6),
            longitude=round(longitude, 6),
            urgency=round(urgency, 2),
            timestamp=int(time.time()),
            sequence=self._sequence_counter,
        )

        encoded = _encode_packet(pkt)
        raw_hex = encoded.hex().upper()

        if num_hops is None:
            num_hops = random.randint(3, 7)

        relay_nodes = _generate_hop_topology(num_hops)
        hop_logs: list[HopLog] = []
        total_latency = 0.0
        hops_completed = 0

        for hop_idx, relay_id in enumerate(relay_nodes):
            # Simulate network gateway handshake latency
            latency = random.uniform(HOP_LATENCY_MS_MIN, HOP_LATENCY_MS_MAX)
            await asyncio.sleep(latency / 10_000)   # scaled down to avoid blocking UI
            total_latency += latency

            hop_result, hop_log = await self._simulate_hop(
                hop_number=hop_idx + 1,
                relay_id=relay_id,
                encoded_packet=encoded,
                latency_ms=latency,
            )
            hop_logs.append(hop_log)

            if hop_result:
                hops_completed += 1
            else:
                # Critical hop failed even after retransmissions
                # Continue trying remaining hops (partial delivery)
                pass

        successful_hops = sum(
            1 for h in hop_logs
            if h.status in ("PASS", "RETRANSMIT_OK")
        )
        delivery_ratio = successful_hops / num_hops if num_hops > 0 else 0.0

        success = delivery_ratio >= 0.5   # majority delivery = success
        final_status = "DELIVERED" if success else "FAILED"

        return TransmissionResult(
            success=success,
            total_hops=num_hops,
            hops_completed=hops_completed,
            hop_logs=hop_logs,
            delivery_ratio=round(delivery_ratio, 3),
            total_latency_ms=round(total_latency, 1),
            raw_payload_hex=raw_hex,
            final_status=final_status,
        )

    # ------------------------------------------------------------------
    # Synchronous wrapper (for non-async callers)
    # ------------------------------------------------------------------

    def transmit_sync(
        self,
        latitude:  float,
        longitude: float,
        urgency:   float,
        num_hops:  Optional[int] = None,
    ) -> TransmissionResult:
        """Run the async transmit coroutine in a synchronous context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Loop closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.transmit(latitude, longitude, urgency, num_hops)
            )
        except RuntimeError:
            # Nest asyncio for Streamlit environments
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            return loop.run_until_complete(
                self.transmit(latitude, longitude, urgency, num_hops)
            )

    # ------------------------------------------------------------------
    # Private per-hop simulation
    # ------------------------------------------------------------------

    async def _simulate_hop(
        self,
        hop_number: int,
        relay_id:   str,
        encoded_packet: bytes,
        latency_ms: float,
    ) -> tuple[bool, HopLog]:
        """
        Simulate a single RF hop with packet loss, corruption, and retransmission.
        """
        rssi         = _simulate_rssi()
        distance     = _simulate_distance()
        retries      = 0
        current_data = encoded_packet

        for attempt in range(MAX_RETRANSMISSIONS + 1):
            # --- Packet loss ---
            if random.random() < PACKET_LOSS_PROBABILITY:
                if attempt < MAX_RETRANSMISSIONS:
                    retries += 1
                    await asyncio.sleep(0.001)
                    continue
                else:
                    # All retransmissions failed
                    log = HopLog(
                        hop_number=hop_number,
                        relay_node_id=relay_id,
                        distance_to_next=distance,
                        latency_ms=latency_ms,
                        status="RETRANSMIT_FAIL",
                        retransmit_count=retries,
                        rssi_dbm=rssi,
                        crc_expected="N/A",
                        crc_received="N/A",
                        notes=f"Packet lost over RF after {retries} retransmission(s). RSSI={rssi} dBm",
                    )
                    return False, log

            # --- Bit-flip corruption ---
            if random.random() < BIT_FLIP_PROBABILITY:
                current_data = _flip_random_bit(current_data)

            # --- CRC validation ---
            valid, exp_crc, recv_crc = _decode_and_verify(current_data)

            if valid:
                status = "PASS" if retries == 0 else "RETRANSMIT_OK"
                notes = (
                    f"Clean delivery. RSSI={rssi} dBm, dist={distance} m"
                    if retries == 0
                    else f"Delivered after {retries} retry(ies). RSSI={rssi} dBm"
                )
                log = HopLog(
                    hop_number=hop_number,
                    relay_node_id=relay_id,
                    distance_to_next=distance,
                    latency_ms=latency_ms,
                    status=status,
                    retransmit_count=retries,
                    rssi_dbm=rssi,
                    crc_expected=exp_crc,
                    crc_received=recv_crc,
                    notes=notes,
                )
                return True, log

            else:
                # Checksum mismatch → request retransmission
                if attempt < MAX_RETRANSMISSIONS:
                    current_data = encoded_packet   # reset to original for retransmit
                    retries += 1
                    await asyncio.sleep(0.001)
                    continue
                else:
                    log = HopLog(
                        hop_number=hop_number,
                        relay_node_id=relay_id,
                        distance_to_next=distance,
                        latency_ms=latency_ms,
                        status="CORRUPTED",
                        retransmit_count=retries,
                        rssi_dbm=rssi,
                        crc_expected=exp_crc,
                        crc_received=recv_crc,
                        notes=(
                            f"⚠ CRC MISMATCH after {retries} retransmit(s). "
                            f"Expected 0x{exp_crc} got 0x{recv_crc}. "
                            f"Bit-flip detected. RSSI={rssi} dBm."
                        ),
                    )
                    return False, log

        # Should never reach here
        return False, HopLog(
            hop_number=hop_number, relay_node_id=relay_id,
            distance_to_next=distance, latency_ms=latency_ms,
            status="LOST", retransmit_count=retries,
            rssi_dbm=rssi, crc_expected="N/A", crc_received="N/A",
            notes="Unknown failure.",
        )
