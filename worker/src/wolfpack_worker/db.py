"""Thin Supabase client wrapper for the worker.

Uses SUPABASE_SERVICE_ROLE_KEY (never the anon key) because the mechanical
worker needs to write raw trades, prices, and trust weights — writes that RLS
denies to the anon role by design (see supabase/migrations/20260920_0001_init_schema.sql).
The Claude rationale-writing step, by contrast, is deliberately scoped to the
anon key plus the write-once `set_trade_rationale` RPC — this module is not
used by that step.
"""

from __future__ import annotations

from supabase import Client, create_client

from wolfpack_worker.config import WorkerConfig


def get_client(config: WorkerConfig) -> Client:
    """Build a Supabase client authenticated with the service-role key.

    This client can write to any table (RLS grants service_role full access)
    and must only ever be constructed inside the worker process — never
    shipped to the frontend or any anon-keyed context.
    """
    if not config.supabase_url or not config.supabase_service_role_key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY — cannot "
            "construct a service-role Supabase client."
        )
    return create_client(config.supabase_url, config.supabase_service_role_key)
