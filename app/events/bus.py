from collections import deque
from queue import (
    Empty,
    Queue,
)
from threading import RLock
from typing import Optional

from app.events.models import (
    EventFilter,
    EventPage,
    EventSubscription,
    NexusEvent,
    SubscriptionNotFoundError,
)


class EventBus:
    """
    Thread-safe event bus for NEXUS.

    Provides:
    - event publishing
    - bounded event history
    - event filtering
    - live subscriptions
    - subscriber queues
    """

    def __init__(
        self,
        max_history: int = 1000,
        subscriber_queue_size: int = 100,
    ):
        if max_history < 1:
            raise ValueError(
                "max_history must be >= 1."
            )

        if subscriber_queue_size < 1:
            raise ValueError(
                "subscriber_queue_size "
                "must be >= 1."
            )

        self.max_history = (
            max_history
        )

        self.subscriber_queue_size = (
            subscriber_queue_size
        )

        self._events: deque[
            NexusEvent
        ] = deque(
            maxlen=max_history
        )

        self._subscriptions: dict[
            str,
            EventSubscription,
        ] = {}

        self._queues: dict[
            str,
            Queue,
        ] = {}

        self._lock = RLock()

    def publish(
        self,
        event: NexusEvent,
    ) -> NexusEvent:
        """
        Publish an event to history and
        matching live subscribers.
        """

        with self._lock:
            self._events.append(
                event
            )

            subscriptions = list(
                self._subscriptions
                .values()
            )

            for subscription in (
                subscriptions
            ):
                if not subscription.active:
                    continue

                if not self._matches_subscription(
                    event,
                    subscription,
                ):
                    continue

                queue = self._queues.get(
                    subscription.id
                )

                if queue is None:
                    continue

                # Keep streaming non-blocking.
                # If a consumer is too slow,
                # discard its oldest event.
                if queue.full():
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass

                queue.put_nowait(
                    event
                )

        return event

    def history(
        self,
        event_filter: Optional[
            EventFilter
        ] = None,
        *,
        limit: Optional[int] = None,
    ) -> EventPage:
        """
        Return retained events matching
        the supplied filter.
        """

        if (
            limit is not None
            and limit < 1
        ):
            raise ValueError(
                "limit must be >= 1."
            )

        with self._lock:
            events = list(
                self._events
            )

        if event_filter is not None:
            events = [
                event
                for event in events
                if self._matches_filter(
                    event,
                    event_filter,
                )
            ]

        total = len(
            events
        )

        if limit is not None:
            events = events[
                -limit:
            ]

        return EventPage(
            events=events,
            count=len(
                events
            ),
            total=total,
        )

    def subscribe(
        self,
        *,
        run_id: Optional[
            str
        ] = None,
        job_id: Optional[
            str
        ] = None,
    ) -> EventSubscription:
        """
        Create a live event subscription.
        """

        subscription = (
            EventSubscription(
                run_id=run_id,
                job_id=job_id,
            )
        )

        queue = Queue(
            maxsize=(
                self.subscriber_queue_size
            )
        )

        with self._lock:
            self._subscriptions[
                subscription.id
            ] = subscription

            self._queues[
                subscription.id
            ] = queue

        return subscription

    def unsubscribe(
        self,
        subscription_id: str,
    ) -> EventSubscription:
        """
        Deactivate and remove a live
        subscription.
        """

        with self._lock:
            subscription = (
                self._subscriptions
                .get(
                    subscription_id
                )
            )

            if subscription is None:
                raise (
                    SubscriptionNotFoundError(
                        "Subscription not found: "
                        f"{subscription_id}"
                    )
                )

            subscription.active = False

            self._subscriptions.pop(
                subscription_id,
                None,
            )

            self._queues.pop(
                subscription_id,
                None,
            )

        return subscription

    def get_subscription(
        self,
        subscription_id: str,
    ) -> EventSubscription:
        with self._lock:
            subscription = (
                self._subscriptions
                .get(
                    subscription_id
                )
            )

            if subscription is None:
                raise (
                    SubscriptionNotFoundError(
                        "Subscription not found: "
                        f"{subscription_id}"
                    )
                )

            return subscription

    def get_event(
        self,
        subscription_id: str,
        *,
        timeout: Optional[
            float
        ] = None,
    ) -> Optional[
        NexusEvent
    ]:
        """
        Wait for the next event for a
        subscription.

        Returns None on timeout.
        """

        with self._lock:
            queue = self._queues.get(
                subscription_id
            )

            if queue is None:
                raise (
                    SubscriptionNotFoundError(
                        "Subscription not found: "
                        f"{subscription_id}"
                    )
                )

        try:
            return queue.get(
                timeout=timeout
            )

        except Empty:
            return None

    def pending_count(
        self,
        subscription_id: str,
    ) -> int:
        with self._lock:
            queue = self._queues.get(
                subscription_id
            )

            if queue is None:
                raise (
                    SubscriptionNotFoundError(
                        "Subscription not found: "
                        f"{subscription_id}"
                    )
                )

            return queue.qsize()

    def subscription_count(
        self,
    ) -> int:
        with self._lock:
            return len(
                self._subscriptions
            )

    def event_count(
        self,
    ) -> int:
        with self._lock:
            return len(
                self._events
            )

    def clear(
        self,
    ) -> None:
        """
        Clear retained history without
        destroying subscriptions.
        """

        with self._lock:
            self._events.clear()

    def _matches_subscription(
        self,
        event: NexusEvent,
        subscription: (
            EventSubscription
        ),
    ) -> bool:
        if (
            subscription.run_id
            is not None
            and event.run_id
            != subscription.run_id
        ):
            return False

        if (
            subscription.job_id
            is not None
            and event.job_id
            != subscription.job_id
        ):
            return False

        return True

    def _matches_filter(
        self,
        event: NexusEvent,
        event_filter: EventFilter,
    ) -> bool:
        if (
            event_filter.run_id
            is not None
            and event.run_id
            != event_filter.run_id
        ):
            return False

        if (
            event_filter.job_id
            is not None
            and event.job_id
            != event_filter.job_id
        ):
            return False

        if (
            event_filter.event_types
            and event.type
            not in event_filter.event_types
        ):
            return False

        if (
            event_filter.severities
            and event.severity
            not in event_filter.severities
        ):
            return False

        return True
