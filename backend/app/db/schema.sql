-- The documented Postgres/Supabase schema (hitl-and-docker.md §2).
--
-- This is the native form to run in the Supabase SQL editor when you point the
-- project at Postgres via DATABASE_URL. You do NOT need to run it for the local
-- SQLite default: `app.db.store.setup()` creates portable equivalents of the
-- `cases` / `case_messages` / `audit_log` tables automatically.
--
-- LangGraph's own checkpoint tables are created by `checkpointer.setup()` (called
-- for you in app/db/checkpointer.py) - do not create those by hand.
--
-- Deviation from the doc's DDL: ids are `text`, not `uuid`. The fixture data uses
-- readable ids (`cust_1001`, `ch_5001`), and `app/db/seed.py` will insert those
-- same rows, so a `uuid` column would not accept them. Documented in PROGRESS.md.

create extension if not exists vector;

-- fixture / demo data
create table if not exists customers (
    id text primary key,
    name text not null,
    email text not null,
    tier text not null default 'standard',
    account_status text not null default 'active'
);

create table if not exists charges (
    id text primary key,
    customer_id text references customers(id),
    order_id text,
    amount numeric(10,2) not null,
    status text not null,              -- succeeded | failed | refunded
    created_at timestamptz not null default now()
);

create table if not exists subscriptions (
    id text primary key,
    customer_id text references customers(id),
    plan text not null,
    status text not null,              -- active | canceled | past_due
    price numeric(10,2) not null,
    renewal_date date
);

-- cases run through the graph
create table if not exists cases (
    id text primary key,
    customer_id text references customers(id),
    category text,
    status text not null default 'in_progress',  -- in_progress|resolved|escalated|awaiting_human
    customer_message text,
    escalation_reason text,
    case_brief jsonb,
    state_snapshot jsonb not null,     -- the whole CaseState (architecture.md §5)
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists cases_status_idx on cases (status);

create table if not exists case_messages (
    case_id text not null references cases(id),
    seq integer not null,
    role text not null,                -- customer | agent | human_reviewer
    content text not null,
    created_at timestamptz not null default now(),
    primary key (case_id, seq)
);

create table if not exists audit_log (
    case_id text not null references cases(id),
    seq integer not null,
    node_name text not null,           -- e.g. 'investigate', 'policy_check'
    event_type text not null,          -- tool_call | llm_call | policy_decision | human_decision
    payload jsonb not null,
    created_at timestamptz not null default now(),
    primary key (case_id, seq)
);

-- knowledge base for RAG (optional; the default build uses local Chroma)
create table if not exists knowledge_chunks (
    id text primary key,
    source text not null,
    content text not null,
    embedding vector(1536)
);

-- eval framework (created by step 4's seed/migration; listed here so one file
-- describes the whole database)
create table if not exists eval_cases (
    id text primary key,
    scenario_name text not null,
    input_message text not null,
    seed_data jsonb not null,
    expected_action text,
    expected_escalation boolean
);

create table if not exists eval_runs (
    id text primary key,
    run_label text not null,
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    metrics jsonb
);

create table if not exists eval_results (
    id text primary key,
    run_id text references eval_runs(id),
    eval_case_id text references eval_cases(id),
    case_id text references cases(id),
    passed boolean,
    notes text
);
