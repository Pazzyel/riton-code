from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import List, Optional
import uuid

from persistence import (
    PersistenceStore,
    SessionRecord,
    get_persistence_store,
    set_persistence_store,
)


logger = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run the coding agent loop."
    )
    session_group: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "-s",
        "--session",
        dest="session_id",
        help="Continue an existing session.",
    )
    session_group.add_argument(
        "-l",
        "--list-sessions",
        action="store_true",
        help="List sessions ordered by their last visible conversation time.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser


def print_sessions(store: PersistenceStore) -> None:
    sessions: List[SessionRecord] = store.list_sessions()
    for session in sessions:
        print(f"{session.session_id} {session.updated_at}")


async def run_session(
    store: PersistenceStore,
    session_id: str,
    resumed: bool,
) -> None:
    """Lazily construct and run a main-agent session."""
    # Keep runtime imports below the list-sessions path so `-l` stays lightweight.
    from session import MainSession

    session: MainSession = await MainSession.create(store, session_id, resumed)
    await session.run()


def main(argv: Optional[List[str]] = None) -> int:
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    store: PersistenceStore = get_persistence_store()
    set_persistence_store(store)

    if args.list_sessions:
        print_sessions(store)
        return 0

    resumed: bool = args.session_id is not None
    if resumed:
        session_id: str = str(args.session_id)
        if not store.session_exists(session_id):
            print(f"Session not found: {session_id}", file=sys.stderr)
            return 2
    else:
        session_id = f"session_{uuid.uuid4().hex}"
        store.create_session(session_id)
        print(f"session_id: {session_id}")

    try:
        asyncio.run(run_session(store, session_id, resumed))
    except (ValueError, TypeError) as exc:
        logger.error("Unable to start session: %s", str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
