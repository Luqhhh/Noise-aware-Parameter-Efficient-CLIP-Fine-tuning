"""Calculate the next five-hour continuation slot for native thread scheduling."""
import argparse
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("Asia/Hong_Kong")
ANCHOR = datetime(2026, 10, 2, 2, 39, tzinfo=ZONE)
INTERVAL = timedelta(hours=5)

def next_slot(now):
    if now.tzinfo is None:
        raise ValueError("The supplied time must include a UTC offset")
    elapsed = now.astimezone(ZONE) - ANCHOR
    return ANCHOR + max(0, elapsed // INTERVAL + 1) * INTERVAL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now", help="Optional offset-aware ISO timestamp")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(ZONE)
    slot = next_slot(now)
    print(json.dumps(dict(automation_id="a", next_hong_kong=slot.isoformat(),
        interval_hours=5, rrule=f"FREQ=DAILY;BYHOUR={slot.hour};BYMINUTE={slot.minute};BYSECOND={slot.second}")))
