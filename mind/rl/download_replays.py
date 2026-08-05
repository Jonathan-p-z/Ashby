"""Downloads ranked 1v1 replays from ballchasing.com for the imitation pipeline.

parse_replays.py needs real human games to learn from -- this is what fetches
them. Filters for ranked 1v1 (ranked-duels) in the diamond-1..supersonic-legend
band: clearly past "still learning the controls", but not so rare that pages
come back empty.

Downloaded files are named <replay_id>.replay, so re-running after a Ctrl+C
(or an HTTP error) just skips whatever's already on disk instead of
re-downloading it -- no separate progress file that can drift out of sync
with the folder.
"""

import os
import time
import argparse

import requests

REPLAYS_DIR = os.path.join(os.path.dirname(__file__), "replays")

API_BASE = "https://ballchasing.com/api"
PLAYLIST = "ranked-duels"        # 1v1 ranked
MIN_RANK = "diamond-1"
MAX_RANK = "supersonic-legend"   # "ssl"
MAX_REPLAY_DATE = "2023-06-01T00:00:00+00:00"  # RFC3339 -- see replay-date-before below
PAGE_SIZE = 200                  # ballchasing's max page size per request
REQUEST_INTERVAL = 1.0           # seconds -- stay at or under 1 req/sec (replay listing)
DOWNLOAD_INTERVAL = 2.0          # seconds -- free accounts cap file downloads at ~30/minute
RATE_LIMIT_RETRY_WAIT = 60.0     # seconds -- backoff on a 429 before retrying the same replay


def _session() -> requests.Session:
    token = os.environ.get("BALLCHASING_TOKEN")
    if not token:
        raise RuntimeError(
            "BALLCHASING_TOKEN is not set. Grab a key from your ballchasing.com "
            "account settings and export it -- the key is read from this "
            "environment variable only, never hardcoded."
        )
    session = requests.Session()
    session.headers.update({"Authorization": token})
    return session


class RateLimiter:
    """Blocks until REQUEST_INTERVAL has passed since the last call went out.

    ballchasing throttles (and can eventually ban) clients that ignore its rate
    limit, and a --count of a few hundred replays means this has to run
    unattended for a while without tripping that.
    """

    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self):
        remaining = self.interval - (time.monotonic() - self._last)
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


def _iter_replay_ids(session: requests.Session, limiter: RateLimiter):
    """Yields replay ids one page at a time via ballchasing's cursor-based
    pagination (a `next` URL in the response body) until pages run out."""
    url = f"{API_BASE}/replays"
    params = {
        "playlist": PLAYLIST,
        "min-rank": MIN_RANK,
        "max-rank": MAX_RANK,
        # boxcars_py chokes on actor types (e.g. TAGame.Default__ViralItemActor_TA,
        # a cosmetic item) introduced by newer game updates -- pre-June-2023 replays
        # predate those and parse cleanly, so filter them out at the source instead
        # of downloading replays parse_replays.py can only skip anyway.
        "replay-date-before": MAX_REPLAY_DATE,
        "count": PAGE_SIZE,
        "sort-by": "replay-date",
        "sort-dir": "desc",
    }
    while url:
        limiter.wait()
        resp = session.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        for entry in payload.get("list", []):
            yield entry["id"]
        url = payload.get("next")
        params = None  # `next` already carries the full query string


def _download_one(session: requests.Session, limiter: RateLimiter, replay_id: str, dest: str):
    # A 429 here means we've already blown past the free-tier download cap --
    # skipping the replay would just permanently lose it, so wait out the
    # window and retry instead of moving on.
    while True:
        limiter.wait()
        resp = session.get(f"{API_BASE}/replays/{replay_id}/file")
        if resp.status_code == 429:
            print(f"\n  rate limited on {replay_id} -- waiting {RATE_LIMIT_RETRY_WAIT:.0f}s and retrying...")
            time.sleep(RATE_LIMIT_RETRY_WAIT)
            continue
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        return


def download(count: int):
    os.makedirs(REPLAYS_DIR, exist_ok=True)
    session = _session()
    limiter = RateLimiter(REQUEST_INTERVAL)
    download_limiter = RateLimiter(DOWNLOAD_INTERVAL)

    already = {os.path.splitext(f)[0] for f in os.listdir(REPLAYS_DIR) if f.endswith(".replay")}
    downloaded = len(already)
    if downloaded:
        print(f"Resuming -- {downloaded} replay(s) already on disk, skipping those.")

    if downloaded >= count:
        print(f"{downloaded}/{count} replays downloaded -- already have enough.")
        return

    for replay_id in _iter_replay_ids(session, limiter):
        if downloaded >= count:
            break
        if replay_id in already:
            continue
        dest = os.path.join(REPLAYS_DIR, f"{replay_id}.replay")
        try:
            _download_one(session, download_limiter, replay_id, dest)
        except requests.HTTPError as e:
            print(f"\n  skipping {replay_id}: {e}")
            continue
        already.add(replay_id)
        downloaded += 1
        print(f"\r{downloaded}/{count} replays downloaded", end="", flush=True)

    print(f"\n{downloaded}/{count} replays downloaded -> {REPLAYS_DIR}")
    if downloaded < count:
        print("Ran out of matching replays on ballchasing before reaching --count.")


def smoke_test():
    """--test: verify the pieces that don't need a live API call -- rate limiter
    timing and resume-by-filename logic -- so this passes in a sandbox without a
    real BALLCHASING_TOKEN or burning API quota. Runs a live ping too, but only
    if a token happens to be set.
    """
    print("Checking rate limiter...")
    limiter = RateLimiter(0.05)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.1, f"rate limiter let 3 calls through in {elapsed:.3f}s, expected >=0.1s"
    print(f"  rate limiter: ok ({elapsed:.3f}s for 3 calls at a 0.05s interval)")

    print("Checking resume-by-filename logic...")
    os.makedirs(REPLAYS_DIR, exist_ok=True)
    probe_id = "__smoke_test_probe__"
    probe_path = os.path.join(REPLAYS_DIR, f"{probe_id}.replay")
    with open(probe_path, "wb") as f:
        f.write(b"fake")
    try:
        on_disk = {os.path.splitext(f)[0] for f in os.listdir(REPLAYS_DIR) if f.endswith(".replay")}
        assert probe_id in on_disk, "resume check didn't pick up an existing .replay file"
        print("  resume logic: ok (existing files are detected and would be skipped)")
    finally:
        os.remove(probe_path)

    if os.environ.get("BALLCHASING_TOKEN"):
        print("BALLCHASING_TOKEN is set -- checking it against a live request...")
        session = _session()
        resp = session.get(f"{API_BASE}/replays", params={"playlist": PLAYLIST, "count": 1})
        resp.raise_for_status()
        print("  live API check: ok (token accepted)")
    else:
        print("  BALLCHASING_TOKEN not set -- skipping the live API check (nothing else needs it).")

    print("download_replays.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ranked 1v1 replays from ballchasing.com")
    parser.add_argument("--count", type=int, default=50, help="how many replays to download (default: 50)")
    parser.add_argument("--test", action="store_true", help="smoke-test rate limiting/resume logic, no download")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        download(args.count)
