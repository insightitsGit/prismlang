"""PrismLang checkpointers for persisting PrismState across graph runs.

Two backends are provided:

JsonFileCheckpointer  — zero extra dependencies, writes one JSON file per
                        checkpoint under a local directory. Good for dev/testing.

PostgresCheckpointer  — stores envelope sequences in a JSONB column. Requires
                        psycopg2-binary. Can share the same Postgres instance as
                        PrismRAG by pointing DATABASE_URL at the same DSN.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.base import CheckpointTuple

from .exceptions import CheckpointerConnectionError, CheckpointerSchemaError


# ------------------------------------------------------------------ #
# JSON file backend                                                    #
# ------------------------------------------------------------------ #

class JsonFileCheckpointer(BaseCheckpointSaver):
    """Persists checkpoints as JSON files under a local directory.

    Directory layout:
        {root}/{thread_id}/{checkpoint_id}.json
    """

    def __init__(self, root: str = ".prismlang_checkpoints") -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str, checkpoint_id: str) -> Path:
        thread_dir = self.root / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        return thread_dir / f"{checkpoint_id}.json"

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            p = self._path(thread_id, checkpoint_id)
            if not p.exists():
                return None
            data = json.loads(p.read_text())
        else:
            # Return the latest checkpoint for this thread
            thread_dir = self.root / thread_id
            if not thread_dir.exists():
                return None
            files = sorted(thread_dir.glob("*.json"))
            if not files:
                return None
            data = json.loads(files[-1].read_text())

        return CheckpointTuple(
            config=config,
            checkpoint=data["checkpoint"],
            metadata=data.get("metadata", {}),
            parent_config=data.get("parent_config"),
        )

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
        thread_dir = self.root / thread_id
        if not thread_dir.exists():
            return
        files = sorted(thread_dir.glob("*.json"), reverse=True)
        count = 0
        for f in files:
            if limit and count >= limit:
                break
            data = json.loads(f.read_text())
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "checkpoint_id": f.stem}},
                checkpoint=data["checkpoint"],
                metadata=data.get("metadata", {}),
                parent_config=data.get("parent_config"),
            )
            count += 1

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        p = self._path(thread_id, checkpoint_id)
        p.write_text(
            json.dumps(
                {
                    "checkpoint": checkpoint,
                    "metadata": metadata,
                    "parent_config": config,
                },
                default=str,
            )
        )
        return {**config, "configurable": {**config["configurable"], "checkpoint_id": checkpoint_id}}

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: list,
        task_id: str,
    ) -> None:
        # For this simple backend writes are committed atomically in put()
        pass


# ------------------------------------------------------------------ #
# PostgreSQL backend                                                   #
# ------------------------------------------------------------------ #

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS prismlang_checkpoint (
    thread_id      TEXT        NOT NULL,
    checkpoint_id  TEXT        NOT NULL,
    tenant_id      TEXT,
    checkpoint     JSONB       NOT NULL,
    metadata       JSONB       NOT NULL DEFAULT '{}',
    parent_config  JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);
CREATE INDEX IF NOT EXISTS ix_prismlang_cp_thread
    ON prismlang_checkpoint (thread_id, created_at DESC);
"""


class PostgresCheckpointer(BaseCheckpointSaver):
    """Persists PrismState checkpoints in a PostgreSQL JSONB column.

    Compatible with PrismRAG's database — point DATABASE_URL at the same DSN.

    Args:
        dsn: PostgreSQL DSN string, e.g. "postgresql://user:pass@localhost/db".
             Falls back to the DATABASE_URL environment variable.
    """

    def __init__(self, dsn: Optional[str] = None) -> None:
        super().__init__()
        try:
            import psycopg2
        except ImportError as exc:
            raise ImportError(
                "psycopg2-binary is required for PostgresCheckpointer. "
                "Install it with: pip install psycopg2-binary"
            ) from exc

        self._dsn = dsn or os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise CheckpointerConnectionError(
                "postgresql", cause=ValueError("No DSN provided. Pass dsn= or set the DATABASE_URL environment variable.")
            )
        self._ensure_schema()

    def _connect(self):
        import psycopg2
        try:
            return psycopg2.connect(self._dsn)
        except Exception as exc:
            raise CheckpointerConnectionError("postgresql", cause=exc) from exc

    def _ensure_schema(self) -> None:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
            conn.commit()
            conn.close()
        except CheckpointerConnectionError:
            raise
        except Exception as exc:
            raise CheckpointerSchemaError(
                f"Failed to create prismlang_checkpoint schema: {exc}"
            ) from exc

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                if checkpoint_id:
                    cur.execute(
                        "SELECT checkpoint, metadata, parent_config FROM prismlang_checkpoint "
                        "WHERE thread_id=%s AND checkpoint_id=%s",
                        (thread_id, checkpoint_id),
                    )
                else:
                    cur.execute(
                        "SELECT checkpoint, metadata, parent_config FROM prismlang_checkpoint "
                        "WHERE thread_id=%s ORDER BY created_at DESC LIMIT 1",
                        (thread_id,),
                    )
                row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return None
        checkpoint, metadata, parent_config = row
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata or {},
            parent_config=parent_config,
        )

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
        q = (
            "SELECT thread_id, checkpoint_id, checkpoint, metadata, parent_config "
            "FROM prismlang_checkpoint WHERE thread_id=%s ORDER BY created_at DESC"
        )
        params = [thread_id]
        if limit:
            q += " LIMIT %s"
            params.append(limit)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(q, params)
                for row in cur.fetchall():
                    tid, cid, checkpoint, metadata, parent_config = row
                    yield CheckpointTuple(
                        config={"configurable": {"thread_id": tid, "checkpoint_id": cid}},
                        checkpoint=checkpoint,
                        metadata=metadata or {},
                        parent_config=parent_config,
                    )
        finally:
            conn.close()

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        tenant_id = checkpoint.get("channel_values", {}).get("tenant_id")
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prismlang_checkpoint
                        (thread_id, checkpoint_id, tenant_id, checkpoint, metadata, parent_config)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                    ON CONFLICT (thread_id, checkpoint_id) DO UPDATE
                        SET checkpoint = EXCLUDED.checkpoint,
                            metadata   = EXCLUDED.metadata
                    """,
                    (
                        thread_id,
                        checkpoint_id,
                        tenant_id,
                        json.dumps(checkpoint, default=str),
                        json.dumps(dict(metadata), default=str),
                        json.dumps(config, default=str),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return {**config, "configurable": {**config["configurable"], "checkpoint_id": checkpoint_id}}

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: list,
        task_id: str,
    ) -> None:
        pass


# ------------------------------------------------------------------ #
# Async JSON file backend                                              #
# ------------------------------------------------------------------ #

class AsyncJsonFileCheckpointer(BaseCheckpointSaver):
    """Async variant of JsonFileCheckpointer using aiofiles.

    Requires: pip install prismlang[async-files]
    """

    def __init__(self, root: str = ".prismlang_checkpoints") -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str, checkpoint_id: str) -> Path:
        thread_dir = self.root / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        return thread_dir / f"{checkpoint_id}.json"

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.aget_tuple(config))
        finally:
            loop.close()

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        try:
            import aiofiles
        except ImportError as exc:
            raise ImportError(
                "aiofiles is required for AsyncJsonFileCheckpointer. "
                "Install with: pip install 'prismlang[async-files]'"
            ) from exc

        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            p = self._path(thread_id, checkpoint_id)
            if not p.exists():
                return None
            async with aiofiles.open(p) as f:
                data = json.loads(await f.read())
        else:
            thread_dir = self.root / thread_id
            if not thread_dir.exists():
                return None
            files = sorted(thread_dir.glob("*.json"))
            if not files:
                return None
            async with aiofiles.open(files[-1]) as f:
                data = json.loads(await f.read())

        return CheckpointTuple(
            config=config,
            checkpoint=data["checkpoint"],
            metadata=data.get("metadata", {}),
            parent_config=data.get("parent_config"),
        )

    def list(self, config, *, filter=None, before=None, limit=None):
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
        thread_dir = self.root / thread_id
        if not thread_dir.exists():
            return
        files = sorted(thread_dir.glob("*.json"), reverse=True)
        count = 0
        for f in files:
            if limit and count >= limit:
                break
            data = json.loads(f.read_text())
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "checkpoint_id": f.stem}},
                checkpoint=data["checkpoint"],
                metadata=data.get("metadata", {}),
                parent_config=data.get("parent_config"),
            )
            count += 1

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            import aiofiles
        except ImportError as exc:
            raise ImportError(
                "aiofiles is required for AsyncJsonFileCheckpointer. "
                "Install with: pip install 'prismlang[async-files]'"
            ) from exc

        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        p = self._path(thread_id, checkpoint_id)
        content = json.dumps(
            {"checkpoint": checkpoint, "metadata": metadata, "parent_config": config},
            default=str,
        )
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        return {**config, "configurable": {**config["configurable"], "checkpoint_id": checkpoint_id}}

    def put(self, config, checkpoint, metadata, new_versions):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.aput(config, checkpoint, metadata, new_versions))
        finally:
            loop.close()

    def put_writes(self, config, writes, task_id):
        pass


# ------------------------------------------------------------------ #
# Async PostgreSQL backend                                             #
# ------------------------------------------------------------------ #

class AsyncPostgresCheckpointer(BaseCheckpointSaver):
    """Async variant of PostgresCheckpointer using asyncpg.

    Requires: pip install prismlang[async-postgres]
    """

    def __init__(self, dsn: Optional[str] = None) -> None:
        super().__init__()
        self._dsn = dsn or os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise CheckpointerConnectionError(
                "asyncpg", cause=ValueError("No DSN provided. Pass dsn= or set DATABASE_URL.")
            )
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as exc:
                raise ImportError(
                    "asyncpg is required for AsyncPostgresCheckpointer. "
                    "Install with: pip install 'prismlang[async-postgres]'"
                ) from exc
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(self._dsn)
                async with self._pool.acquire() as conn:
                    await conn.execute(_CREATE_TABLE)
            except Exception as exc:
                raise CheckpointerConnectionError("asyncpg", cause=exc) from exc
        return self._pool

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        pool = await self._get_pool()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        async with pool.acquire() as conn:
            if checkpoint_id:
                row = await conn.fetchrow(
                    "SELECT checkpoint, metadata, parent_config FROM prismlang_checkpoint "
                    "WHERE thread_id=$1 AND checkpoint_id=$2",
                    thread_id, checkpoint_id,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT checkpoint, metadata, parent_config FROM prismlang_checkpoint "
                    "WHERE thread_id=$1 ORDER BY created_at DESC LIMIT 1",
                    thread_id,
                )

        if not row:
            return None
        return CheckpointTuple(
            config=config,
            checkpoint=json.loads(row["checkpoint"]),
            metadata=json.loads(row["metadata"] or "{}"),
            parent_config=json.loads(row["parent_config"]) if row["parent_config"] else None,
        )

    def get_tuple(self, config):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.aget_tuple(config))
        finally:
            loop.close()

    def list(self, config, *, filter=None, before=None, limit=None):
        return iter([])  # use alist() for async listing

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        pool = await self._get_pool()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        tenant_id = checkpoint.get("channel_values", {}).get("tenant_id")

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO prismlang_checkpoint
                    (thread_id, checkpoint_id, tenant_id, checkpoint, metadata, parent_config)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb)
                ON CONFLICT (thread_id, checkpoint_id) DO UPDATE
                    SET checkpoint = EXCLUDED.checkpoint,
                        metadata   = EXCLUDED.metadata
                """,
                thread_id,
                checkpoint_id,
                tenant_id,
                json.dumps(checkpoint, default=str),
                json.dumps(dict(metadata), default=str),
                json.dumps(config, default=str),
            )

        return {**config, "configurable": {**config["configurable"], "checkpoint_id": checkpoint_id}}

    def put(self, config, checkpoint, metadata, new_versions):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.aput(config, checkpoint, metadata, new_versions))
        finally:
            loop.close()

    def put_writes(self, config, writes, task_id):
        pass
