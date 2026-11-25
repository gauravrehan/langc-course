"""Notification delivery.

The agent builds a channel-neutral `Notification` and hands it to a
`Notifier`. For now `LogNotifier` only logs it; the notification is also
stored with the intervention record. A future TeamsNotifier would map the
same payload to an Adaptive Card without any graph changes.
"""

import logging
from abc import ABC, abstractmethod

from pat.models import Notification

logger = logging.getLogger("pat")


class Notifier(ABC):
    @abstractmethod
    def send(self, notification: Notification) -> None: ...


class LogNotifier(Notifier):
    def send(self, notification: Notification) -> None:
        logger.info(
            "Notification for %s to %s (%s): %s",
            notification.pat_id,
            notification.recipient,
            notification.urgency,
            notification.title,
        )


class RecordingNotifier(Notifier):
    """Keeps notifications in a list. Used by tests and demos."""

    def __init__(self) -> None:
        self.sent: list[Notification] = []

    def send(self, notification: Notification) -> None:
        self.sent.append(notification)
