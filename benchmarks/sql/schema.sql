-- PrismLang Benchmark Schema
-- Database: prismLangDB
-- Three domains: healthcare, finance, trade_market
-- Plus benchmark results tracking

-- =========================================================
-- SHARED: benchmark results
-- =========================================================
CREATE SCHEMA IF NOT EXISTS bench;

CREATE TABLE IF NOT EXISTS bench.run_results (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT        NOT NULL,           -- UUID per benchmark run
    domain          TEXT        NOT NULL,           -- healthcare | finance | trade_market
    mode            TEXT        NOT NULL,           -- standard | prismlang
    model           TEXT        NOT NULL,           -- gemini-2.0-flash | mock
    tenant_id       TEXT,
    turns           INT         NOT NULL,
    -- Time metrics
    total_ms        FLOAT       NOT NULL,           -- end-to-end wall clock
    llm_ms          FLOAT       NOT NULL,           -- cumulative LLM call time
    encode_ms       FLOAT       NOT NULL,           -- cumulative PrismLang encode time (0 for standard)
    -- Token metrics
    prompt_tokens   INT         NOT NULL,           -- total input tokens across all turns
    output_tokens   INT         NOT NULL,           -- total output tokens across all turns
    -- Payload metrics
    state_bytes     INT         NOT NULL,           -- total bytes in state at end of graph
    vector_bytes    INT         NOT NULL,           -- bytes if PrismLang vectors only (0 for standard)
    compression_ratio FLOAT     GENERATED ALWAYS AS (
        CASE WHEN vector_bytes > 0 THEN state_bytes::float / NULLIF(vector_bytes, 0) ELSE NULL END
    ) STORED,
    -- System metrics
    peak_rss_mb     FLOAT       NOT NULL,           -- peak resident set size in MB
    -- Audit
    category_flow   TEXT[],                         -- sequence of category slugs (PrismLang only)
    tenant_matrix_fp TEXT,                          -- JL matrix fingerprint
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_bench_run_id    ON bench.run_results(run_id);
CREATE INDEX IF NOT EXISTS ix_bench_domain    ON bench.run_results(domain, mode);
CREATE INDEX IF NOT EXISTS ix_bench_created   ON bench.run_results(created_at DESC);


-- =========================================================
-- DOMAIN 1: HEALTHCARE
-- =========================================================
CREATE SCHEMA IF NOT EXISTS healthcare;

CREATE TABLE IF NOT EXISTS healthcare.patients (
    id              SERIAL PRIMARY KEY,
    mrn             TEXT        UNIQUE NOT NULL,    -- Medical Record Number
    full_name       TEXT        NOT NULL,
    dob             DATE        NOT NULL,
    gender          TEXT,
    blood_type      TEXT,
    allergies       TEXT[],
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS healthcare.clinical_notes (
    id              SERIAL PRIMARY KEY,
    patient_mrn     TEXT        REFERENCES healthcare.patients(mrn),
    note_type       TEXT        NOT NULL,           -- admission | progress | discharge | radiology | lab
    authored_by     TEXT        NOT NULL,           -- physician name
    note_text       TEXT        NOT NULL,
    note_date       DATE        NOT NULL,
    icd10_codes     TEXT[],
    embedding       vector(384),                    -- PrismLang encoded vector
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS healthcare.diagnoses (
    id              SERIAL PRIMARY KEY,
    patient_mrn     TEXT        REFERENCES healthcare.patients(mrn),
    icd10_code      TEXT        NOT NULL,
    description     TEXT        NOT NULL,
    severity        TEXT,                           -- mild | moderate | severe | critical
    diagnosed_on    DATE,
    status          TEXT        DEFAULT 'active'    -- active | resolved | chronic
);

CREATE TABLE IF NOT EXISTS healthcare.lab_results (
    id              SERIAL PRIMARY KEY,
    patient_mrn     TEXT        REFERENCES healthcare.patients(mrn),
    test_name       TEXT        NOT NULL,
    result_value    FLOAT,
    result_unit     TEXT,
    reference_range TEXT,
    flag            TEXT,                           -- normal | high | low | critical
    collected_on    DATE        NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_hc_notes_patient ON healthcare.clinical_notes(patient_mrn);
CREATE INDEX IF NOT EXISTS ix_hc_notes_type    ON healthcare.clinical_notes(note_type);
CREATE INDEX IF NOT EXISTS ix_hc_dx_patient    ON healthcare.diagnoses(patient_mrn);


-- =========================================================
-- DOMAIN 2: FINANCE
-- =========================================================
CREATE SCHEMA IF NOT EXISTS finance;

CREATE TABLE IF NOT EXISTS finance.accounts (
    id              SERIAL PRIMARY KEY,
    account_id      TEXT        UNIQUE NOT NULL,
    client_name     TEXT        NOT NULL,
    account_type    TEXT        NOT NULL,           -- individual | institutional | hedge_fund
    aum_usd         FLOAT,
    risk_profile    TEXT,                           -- conservative | moderate | aggressive
    manager         TEXT,
    opened_on       DATE,
    kyc_status      TEXT        DEFAULT 'verified'
);

CREATE TABLE IF NOT EXISTS finance.positions (
    id              SERIAL PRIMARY KEY,
    account_id      TEXT        REFERENCES finance.accounts(account_id),
    ticker          TEXT        NOT NULL,
    asset_class     TEXT        NOT NULL,           -- equity | bond | fx | commodity | derivative
    quantity        FLOAT       NOT NULL,
    avg_cost        FLOAT       NOT NULL,
    current_price   FLOAT       NOT NULL,
    unrealised_pnl  FLOAT       GENERATED ALWAYS AS ((current_price - avg_cost) * quantity) STORED,
    currency        TEXT        DEFAULT 'USD',
    as_of_date      DATE        NOT NULL
);

CREATE TABLE IF NOT EXISTS finance.risk_events (
    id              SERIAL PRIMARY KEY,
    account_id      TEXT        REFERENCES finance.accounts(account_id),
    event_type      TEXT        NOT NULL,           -- var_breach | margin_call | credit_downgrade | concentration
    severity        TEXT        NOT NULL,           -- low | medium | high | critical
    description     TEXT        NOT NULL,
    triggered_at    TIMESTAMPTZ NOT NULL,
    resolved_at     TIMESTAMPTZ,
    embedding       vector(384)
);

CREATE TABLE IF NOT EXISTS finance.transactions (
    id              SERIAL PRIMARY KEY,
    account_id      TEXT        REFERENCES finance.accounts(account_id),
    trade_date      DATE        NOT NULL,
    settle_date     DATE,
    ticker          TEXT        NOT NULL,
    direction       TEXT        NOT NULL,           -- buy | sell | short | cover
    quantity        FLOAT       NOT NULL,
    price           FLOAT       NOT NULL,
    notional_usd    FLOAT       GENERATED ALWAYS AS (quantity * price) STORED,
    status          TEXT        DEFAULT 'settled'
);

CREATE INDEX IF NOT EXISTS ix_fin_pos_account  ON finance.positions(account_id);
CREATE INDEX IF NOT EXISTS ix_fin_risk_account ON finance.risk_events(account_id);
CREATE INDEX IF NOT EXISTS ix_fin_tx_account   ON finance.transactions(account_id, trade_date DESC);


-- =========================================================
-- DOMAIN 3: TRADE MARKET
-- =========================================================
CREATE SCHEMA IF NOT EXISTS trade_market;

CREATE TABLE IF NOT EXISTS trade_market.instruments (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT        UNIQUE NOT NULL,
    name            TEXT        NOT NULL,
    asset_class     TEXT        NOT NULL,           -- equity | etf | future | option | crypto
    exchange        TEXT        NOT NULL,
    sector          TEXT,
    currency        TEXT        DEFAULT 'USD'
);

CREATE TABLE IF NOT EXISTS trade_market.ohlcv (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT        REFERENCES trade_market.instruments(symbol),
    trade_date      DATE        NOT NULL,
    open_price      FLOAT       NOT NULL,
    high_price      FLOAT       NOT NULL,
    low_price       FLOAT       NOT NULL,
    close_price     FLOAT       NOT NULL,
    volume          BIGINT      NOT NULL,
    vwap            FLOAT,
    UNIQUE(symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS trade_market.order_book_snapshots (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT        REFERENCES trade_market.instruments(symbol),
    snapshot_ts     TIMESTAMPTZ NOT NULL,
    bid_price       FLOAT,
    ask_price       FLOAT,
    bid_size        FLOAT,
    ask_size        FLOAT,
    spread_bps      FLOAT       GENERATED ALWAYS AS (
        CASE WHEN bid_price > 0 THEN ((ask_price - bid_price) / bid_price) * 10000 ELSE NULL END
    ) STORED
);

CREATE TABLE IF NOT EXISTS trade_market.signals (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT        REFERENCES trade_market.instruments(symbol),
    signal_date     DATE        NOT NULL,
    signal_type     TEXT        NOT NULL,           -- momentum | mean_reversion | breakout | sentiment
    direction       TEXT        NOT NULL,           -- long | short | neutral
    confidence      FLOAT,                          -- 0.0 - 1.0
    description     TEXT        NOT NULL,
    embedding       vector(384)
);

CREATE INDEX IF NOT EXISTS ix_tm_ohlcv_sym    ON trade_market.ohlcv(symbol, trade_date DESC);
CREATE INDEX IF NOT EXISTS ix_tm_obs_sym      ON trade_market.order_book_snapshots(symbol, snapshot_ts DESC);
CREATE INDEX IF NOT EXISTS ix_tm_sig_sym      ON trade_market.signals(symbol, signal_date DESC);
