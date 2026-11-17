"""
IndusPredict AI - Industrial MQTT Telemetry Ingestion Client.
Provides read-only telemetry subscription using Paho MQTT with automatic reconnection,
JSON validation, and Qt signal emission.
"""

from datetime import datetime
import json
import socket
from typing import Any, Dict, Optional
import paho.mqtt.client as mqtt
from PySide6.QtCore import QObject, Signal

from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now

logger = get_logger("iot.mqtt")


class MQTTTelemetryClient(QObject):
    """
    Read-only MQTT client subscribing to factory plant sensor streams.
    Strictly isolated: publishes no control or override commands.
    """

    # Signals for UI and TelemetryService
    connected_signal = Signal(str, int)  # broker_host, broker_port
    disconnected_signal = Signal(str)    # reason
    telemetry_received = Signal(dict)    # validated telemetry payload
    error_signal = Signal(str)          # error message

    DEFAULT_BROKER = "localhost"
    DEFAULT_PORT = 1883
    DEFAULT_TOPIC = "induspredict/+/telemetry"

    def __init__(self, client_id: str = "IndusPredict-Desktop-Subscriber", parent=None):
        super().__init__(parent)
        self.client_id = client_id
        self.broker_host = self.DEFAULT_BROKER
        self.broker_port = self.DEFAULT_PORT
        self.topic = self.DEFAULT_TOPIC
        self.is_connected = False

        # Initialize Paho MQTT Client (v2 callback API compatibility)
        try:
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
        except (AttributeError, TypeError):
            # Fallback for older Paho MQTT 1.x
            self.client = mqtt.Client(client_id=self.client_id)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def connect_broker(self, host: str = "localhost", port: int = 1883, topic: str = "induspredict/+/telemetry") -> bool:
        """Initiate background read-only connection to MQTT broker."""
        self.broker_host = host.strip() or self.DEFAULT_BROKER
        self.broker_port = int(port)
        self.topic = topic.strip() or self.DEFAULT_TOPIC

        try:
            logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}...")
            self.client.connect_async(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            err = f"Failed to initiate MQTT connection to {self.broker_host}:{self.broker_port} - {e}"
            logger.error(err)
            self.error_signal.emit(err)
            return False

    def disconnect_broker(self) -> None:
        """Disconnect safely from broker."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
            logger.info("MQTT client disconnected cleanly.")
        except Exception as e:
            logger.error(f"Error disconnecting MQTT client: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        """Callback triggered on successful broker connection."""
        if rc == 0 or (hasattr(rc, "is_failure") and not rc.is_failure):
            self.is_connected = True
            logger.info(f"Connected to MQTT broker. Subscribing read-only to topic: {self.topic}")
            client.subscribe(self.topic, qos=1)
            self.connected_signal.emit(self.broker_host, self.broker_port)
        else:
            err_msg = f"MQTT connection refused with code: {rc}"
            logger.warning(err_msg)
            self.error_signal.emit(err_msg)

    def _on_disconnect(self, client, userdata, flags_or_rc, rc_or_props=None, properties=None) -> None:
        """Callback triggered on broker disconnection."""
        self.is_connected = False
        logger.info("MQTT connection dropped.")
        self.disconnected_signal.emit("Broker connection dropped.")

    def _on_message(self, client, userdata, message) -> None:
        """Parse incoming JSON telemetry payload and emit structured signal."""
        try:
            raw_payload = message.payload.decode("utf-8")
            data = json.loads(raw_payload)

            # Ensure essential keys exist
            machine_code = data.get("machine_code")
            if not machine_code:
                # Try inferring from topic e.g. induspredict/CMP-01/telemetry
                topic_parts = message.topic.split("/")
                if len(topic_parts) >= 2:
                    machine_code = topic_parts[1]
                else:
                    machine_code = "UNKNOWN"

            timestamp_str = data.get("timestamp", utc_now().strftime("%Y-%m-%d %H:%M:%S"))

            # Format validated frame
            payload = {
                "machine_code": machine_code,
                "timestamp": timestamp_str,
                "source": "mqtt",  # Explicitly identified
                "topic": message.topic,
                "sensors": data.get("sensors", {}),
            }

            self.telemetry_received.emit(payload)
        except Exception as e:
            logger.error(f"Error decoding incoming MQTT message on {message.topic}: {e}")
