-- Idempotency support for the mechanical trade-execution worker.
--
-- client_order_id lets the worker's execution.py deterministically derive
-- the same id for a given (persona, ticker, signal date) and use it to
-- recognize a broker order it already submitted (e.g. after a crash between
-- "order accepted" and "broker_order_id recorded").
--
-- (persona_id, ticker, signal_ts) is the true idempotency key: the worker
-- upserts the pending trades row on this constraint with
-- ignore_duplicates=True, so re-running the same signal never inserts a
-- second row (and therefore never places a second broker order).
alter table trades add column client_order_id text;
alter table trades add constraint trades_client_order_id_key unique (client_order_id);
alter table trades add constraint trades_persona_ticker_signal_key unique (persona_id, ticker, signal_ts);
