from __future__ import annotations

from datetime import datetime
from math import floor
from typing import List, Dict, Tuple

from app.core.schemas import UserProfile, Conversation




def get_first_session_date(user: UserProfile) -> datetime:
    """
    Returns the timestamp of the user's first conversation.
    Assumes conversations are already sorted.
    """
    if not user.conversations:
        raise ValueError(f"{user.user_id} has no conversations")

    return user.conversations[0].timestamp


def get_last_session_date(user: UserProfile) -> datetime:
    """
    Returns last conversation timestamp.
    """
    if not user.conversations:
        raise ValueError(f"{user.user_id} has no conversations")

    return user.conversations[-1].timestamp




def compute_week_index(
    event_time: datetime,
    baseline_time: datetime
) -> int:
    """
    Converts datetime into relative week number.

    Week 1 = first 7 days from baseline
    Week 2 = days 7-13
    etc.
    """
    delta_days = (event_time - baseline_time).days
    return floor(delta_days / 7) + 1


def attach_week_indices(user: UserProfile) -> List[Dict]:
    """
    Returns structured conversation timeline with week numbers.
    """

    baseline = get_first_session_date(user)

    timeline = []

    for convo in user.conversations:
        week = compute_week_index(convo.timestamp, baseline)

        timeline.append(
            {
                "user_id": user.user_id,
                "session_id": convo.session_id,
                "timestamp": convo.timestamp,
                "week_index": week,
                "severity": convo.severity.value,
                "tags": convo.tags,
                "message": convo.user_message,
            }
        )

    return timeline




def group_sessions_by_week(user: UserProfile) -> Dict[int, List[Conversation]]:
    """
    Returns:
    {
        1: [conv1, conv2],
        2: [conv3]
    }
    """

    baseline = get_first_session_date(user)
    grouped: Dict[int, List[Conversation]] = {}

    for convo in user.conversations:
        week = compute_week_index(convo.timestamp, baseline)

        if week not in grouped:
            grouped[week] = []

        grouped[week].append(convo)

    return grouped


def get_sessions_in_week(
    user: UserProfile,
    week_index: int
) -> List[Conversation]:
    grouped = group_sessions_by_week(user)
    return grouped.get(week_index, [])




def weeks_between(
    earlier: datetime,
    later: datetime
) -> int:
    """
    Number of weeks between two timestamps.
    """
    delta_days = (later - earlier).days
    return max(0, floor(delta_days / 7))


def days_between(
    earlier: datetime,
    later: datetime
) -> int:
    return max(0, (later - earlier).days)




def sessions_between_dates(
    user: UserProfile,
    start: datetime,
    end: datetime
) -> List[Conversation]:
    """
    Inclusive range filter.
    """
    return [
        c for c in user.conversations
        if start <= c.timestamp <= end
    ]


def recent_sessions(
    user: UserProfile,
    before_time: datetime,
    lookback_days: int = 14
) -> List[Conversation]:
    """
    Sessions in last N days before timestamp.
    """
    from datetime import timedelta

    start = before_time - timedelta(days=lookback_days)

    return [
        c for c in user.conversations
        if start <= c.timestamp < before_time
    ]




def recurring_week_pattern(
    week_indices: List[int]
) -> Dict:
    """
    Detect repeated same-week occurrences.
    Example:
    [2, 6, 10] => every 4 weeks pattern
    """
    if len(week_indices) < 3:
        return {"is_pattern": False}

    gaps = [
        week_indices[i] - week_indices[i - 1]
        for i in range(1, len(week_indices))
    ]

    same_gap = len(set(gaps)) == 1

    return {
        "is_pattern": same_gap,
        "gaps": gaps,
        "average_gap": sum(gaps) / len(gaps),
    }



def build_user_timeline_summary(user: UserProfile) -> Dict:
    """
    Useful for dashboards / debugging.
    """

    first_date = get_first_session_date(user)
    last_date = get_last_session_date(user)

    total_days = (last_date - first_date).days + 1
    total_weeks = compute_week_index(last_date, first_date)

    return {
        "user_id": user.user_id,
        "name": user.name,
        "sessions": len(user.conversations),
        "start_date": first_date.isoformat(),
        "end_date": last_date.isoformat(),
        "span_days": total_days,
        "span_weeks": total_weeks,
    }



if __name__ == "__main__":
    from app.core.loader import load_dataset, get_user

    ds = load_dataset("data/askfirst_synthetic_dataset.json")
    user = get_user(ds, "USR002")

    print(build_user_timeline_summary(user))
    print(attach_week_indices(user))
