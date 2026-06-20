-- PrismLang Benchmark Seed Data
-- Realistic sample rows for all three domains

-- =========================================================
-- HEALTHCARE
-- =========================================================

INSERT INTO healthcare.patients (mrn, full_name, dob, gender, blood_type, allergies) VALUES
('MRN-001', 'James Whitfield',    '1952-03-14', 'M', 'A+',  ARRAY['penicillin','sulfa']),
('MRN-002', 'Sandra Chen',        '1968-07-22', 'F', 'O-',  ARRAY['aspirin']),
('MRN-003', 'Robert Kowalski',    '1975-11-05', 'M', 'B+',  ARRAY[]::TEXT[]),
('MRN-004', 'Maria Delgado',      '1989-02-18', 'F', 'AB+', ARRAY['latex','codeine']),
('MRN-005', 'Thomas Okafor',      '1943-09-30', 'M', 'O+',  ARRAY['ibuprofen'])
ON CONFLICT DO NOTHING;

INSERT INTO healthcare.diagnoses (patient_mrn, icd10_code, description, severity, diagnosed_on, status) VALUES
('MRN-001', 'I25.10', 'Coronary artery disease, unspecified',          'severe',   '2019-04-10', 'chronic'),
('MRN-001', 'E11.9',  'Type 2 diabetes mellitus without complications', 'moderate', '2015-06-22', 'chronic'),
('MRN-001', 'I10',    'Essential (primary) hypertension',               'moderate', '2012-01-15', 'chronic'),
('MRN-002', 'C50.911','Malignant neoplasm of breast, right',            'severe',   '2023-03-01', 'active'),
('MRN-002', 'Z23',    'Encounter for immunization',                     'mild',     '2024-01-10', 'resolved'),
('MRN-003', 'M54.5',  'Low back pain',                                  'mild',     '2023-09-15', 'active'),
('MRN-003', 'J45.20', 'Mild intermittent asthma, uncomplicated',        'mild',     '2010-05-08', 'chronic'),
('MRN-004', 'O34.219','Maternal care for unspecified type of scar',     'moderate', '2024-05-01', 'active'),
('MRN-005', 'J44.1',  'COPD with acute exacerbation',                  'critical', '2024-11-20', 'active'),
('MRN-005', 'I50.9',  'Heart failure, unspecified',                     'severe',   '2023-08-14', 'chronic')
ON CONFLICT DO NOTHING;

INSERT INTO healthcare.lab_results (patient_mrn, test_name, result_value, result_unit, reference_range, flag, collected_on) VALUES
('MRN-001', 'HbA1c',         8.2,  '%',       '4.0-5.6',    'high',     '2024-10-01'),
('MRN-001', 'LDL Cholesterol', 142, 'mg/dL',  '<100',       'high',     '2024-10-01'),
('MRN-001', 'eGFR',          58,   'mL/min',  '>60',        'low',      '2024-10-01'),
('MRN-002', 'CA 15-3',       85,   'U/mL',    '<30',        'critical', '2024-08-15'),
('MRN-002', 'CEA',           12.4, 'ng/mL',   '<2.5',       'high',     '2024-08-15'),
('MRN-003', 'FEV1',          82,   '%pred',   '>80',        'normal',   '2024-09-10'),
('MRN-003', 'CRP',           2.1,  'mg/L',    '<1.0',       'high',     '2024-09-10'),
('MRN-004', 'Hemoglobin',    10.8, 'g/dL',    '12.0-16.0',  'low',      '2024-06-01'),
('MRN-005', 'SpO2',          88,   '%',       '95-100',     'critical', '2024-11-20'),
('MRN-005', 'BNP',           980,  'pg/mL',   '<100',       'critical', '2024-11-20')
ON CONFLICT DO NOTHING;

INSERT INTO healthcare.clinical_notes (patient_mrn, note_type, authored_by, note_text, note_date, icd10_codes) VALUES
('MRN-001', 'progress', 'Dr. Amelia Torres',
 'Patient presents with worsening chest pain on exertion. EKG shows ST changes in V3-V5. Troponin trending upward at 0.08 ng/mL. HbA1c remains suboptimally controlled at 8.2%. Plan: admit for observation, cardiology consult, adjust metformin dose, repeat troponin in 6h. Patient is allergic to penicillin — avoid beta-lactams.',
 '2024-10-01', ARRAY['I25.10','E11.9']),
('MRN-002', 'admission', 'Dr. Kevin Marsh',
 'New diagnosis of right breast malignancy confirmed by core biopsy. ER/PR positive, HER2 negative, grade 2. CA 15-3 elevated at 85 U/mL. Oncology team recommends neoadjuvant chemotherapy followed by lumpectomy. Genetic counselling referral placed. Patient distressed — social work consulted.',
 '2024-08-15', ARRAY['C50.911']),
('MRN-005', 'admission', 'Dr. Priya Nair',
 'Thomas Okafor admitted in acute respiratory distress. SpO2 88% on room air, BNP 980 pg/mL indicating acute decompensated heart failure concurrent with COPD exacerbation. CXR shows bilateral infiltrates and cardiomegaly. Placed on BiPAP, IV furosemide initiated. ICU transfer considered. Ibuprofen allergy documented.',
 '2024-11-20', ARRAY['J44.1','I50.9'])
ON CONFLICT DO NOTHING;

-- =========================================================
-- FINANCE
-- =========================================================

INSERT INTO finance.accounts (account_id, client_name, account_type, aum_usd, risk_profile, manager, opened_on) VALUES
('ACC-F001', 'Meridian Capital Partners',  'institutional', 850000000, 'moderate',     'Julia Hartmann',  '2018-03-01'),
('ACC-F002', 'Northgate Family Office',    'individual',    42000000,  'conservative', 'David Lim',       '2020-09-15'),
('ACC-F003', 'Apex Macro Hedge Fund',      'hedge_fund',    1200000000,'aggressive',   'Rachel Stone',    '2016-06-01'),
('ACC-F004', 'Clearwater Pension Trust',   'institutional', 320000000, 'conservative', 'Marcus Webb',     '2010-01-12'),
('ACC-F005', 'Ironwood Quantitative LLC',  'hedge_fund',    95000000,  'aggressive',   'Sophia Park',     '2022-04-01')
ON CONFLICT DO NOTHING;

INSERT INTO finance.positions (account_id, ticker, asset_class, quantity, avg_cost, current_price, currency, as_of_date) VALUES
('ACC-F001', 'AAPL',   'equity',     50000, 145.20, 189.40, 'USD', '2024-11-01'),
('ACC-F001', 'TLT',    'bond',       200000, 98.50,  92.10, 'USD', '2024-11-01'),
('ACC-F001', 'GLD',    'commodity',  80000,  175.00, 188.50,'USD', '2024-11-01'),
('ACC-F002', 'BRK.B',  'equity',     5000,  310.00,  352.00,'USD', '2024-11-01'),
('ACC-F002', 'AGG',    'bond',       25000,  98.00,   96.50, 'USD', '2024-11-01'),
('ACC-F003', 'SPY',    'equity',     -100000,520.00, 545.00,'USD', '2024-11-01'),
('ACC-F003', 'TLT',    'bond',       500000, 95.00,   92.10, 'USD', '2024-11-01'),
('ACC-F003', 'EURUSD', 'fx',         10000000, 1.05,  1.0820,'USD','2024-11-01'),
('ACC-F004', 'VTI',    'equity',     120000, 220.00,  248.00,'USD', '2024-11-01'),
('ACC-F005', 'QQQ',    'equity',     -50000, 480.00,  495.00,'USD', '2024-11-01')
ON CONFLICT DO NOTHING;

INSERT INTO finance.risk_events (account_id, event_type, severity, description, triggered_at) VALUES
('ACC-F001', 'concentration',   'medium',
 'GLD position exceeds 12% of portfolio NAV. Concentration limit is 10%. Requires rebalancing or risk committee waiver within 5 business days.',
 '2024-11-01 09:15:00+00'),
('ACC-F003', 'var_breach',      'high',
 '1-day 99% VaR has breached the $18M limit at $22.4M following SPY short squeeze. Margin requirements increased by $4.2M. Prime broker notified. Immediate risk review required.',
 '2024-11-01 14:30:00+00'),
('ACC-F003', 'margin_call',     'critical',
 'Prime broker Goldwick issued margin call of $8.7M due to combined losses on SPY short and TLT long position drawdown. Settlement required by COB tomorrow.',
 '2024-11-01 16:45:00+00'),
('ACC-F004', 'credit_downgrade','medium',
 'Moody''s downgraded AGG constituent holding (CORP-2031 bond) from Baa2 to Ba1 (junk). Pension mandate prohibits sub-investment-grade holdings. Forced sale required.',
 '2024-10-28 08:00:00+00'),
('ACC-F005', 'var_breach',      'high',
 'QQQ short position VaR at $3.1M against $2.5M limit. Quant model underestimated implied volatility surge. Risk factor review initiated.',
 '2024-11-01 11:00:00+00')
ON CONFLICT DO NOTHING;

INSERT INTO finance.transactions (account_id, trade_date, settle_date, ticker, direction, quantity, price) VALUES
('ACC-F001', '2024-10-30', '2024-11-01', 'AAPL',   'buy',   10000, 188.50),
('ACC-F001', '2024-10-30', '2024-11-01', 'GLD',    'buy',   5000,  187.00),
('ACC-F003', '2024-10-29', '2024-10-31', 'SPY',    'short', 20000, 528.00),
('ACC-F003', '2024-10-29', '2024-10-31', 'TLT',    'buy',   50000,  94.20),
('ACC-F004', '2024-10-31', '2024-11-04', 'VTI',    'buy',   5000,  245.00),
('ACC-F005', '2024-11-01', '2024-11-05', 'QQQ',    'short', 5000,  488.00)
ON CONFLICT DO NOTHING;

-- =========================================================
-- TRADE MARKET
-- =========================================================

INSERT INTO trade_market.instruments (symbol, name, asset_class, exchange, sector, currency) VALUES
('SPY',    'SPDR S&P 500 ETF Trust',              'etf',    'NYSE',   'broad_market', 'USD'),
('QQQ',    'Invesco QQQ Trust',                   'etf',    'NASDAQ', 'technology',   'USD'),
('NVDA',   'NVIDIA Corporation',                  'equity', 'NASDAQ', 'technology',   'USD'),
('AAPL',   'Apple Inc.',                          'equity', 'NASDAQ', 'technology',   'USD'),
('GLD',    'SPDR Gold Shares ETF',                'etf',    'NYSE',   'commodity',    'USD'),
('TLT',    'iShares 20+ Year Treasury Bond ETF',  'etf',    'NYSE',   'fixed_income', 'USD'),
('EURUSD', 'Euro / US Dollar',                    'future', 'CME',    'fx',           'USD'),
('CL',     'Crude Oil Futures (WTI)',              'future', 'NYMEX',  'commodity',    'USD'),
('BTC',    'Bitcoin',                             'crypto', 'CBOE',   'crypto',       'USD'),
('TSLA',   'Tesla Inc.',                          'equity', 'NASDAQ', 'automotive',   'USD')
ON CONFLICT DO NOTHING;

INSERT INTO trade_market.ohlcv (symbol, trade_date, open_price, high_price, low_price, close_price, volume, vwap) VALUES
('NVDA',  '2024-10-28', 138.50, 144.20, 137.80, 143.90, 52000000, 141.85),
('NVDA',  '2024-10-29', 144.00, 148.50, 143.20, 147.80, 61000000, 146.10),
('NVDA',  '2024-10-30', 147.50, 152.10, 146.80, 151.20, 58000000, 149.70),
('NVDA',  '2024-10-31', 151.00, 153.40, 148.90, 149.50, 45000000, 151.00),
('NVDA',  '2024-11-01', 149.50, 155.80, 149.20, 154.90, 67000000, 152.80),
('SPY',   '2024-10-28', 538.00, 542.50, 536.20, 541.80, 72000000, 539.90),
('SPY',   '2024-10-29', 541.50, 545.20, 540.80, 543.90, 68000000, 543.10),
('SPY',   '2024-10-30', 543.50, 547.00, 542.10, 545.60, 65000000, 544.80),
('SPY',   '2024-10-31', 545.00, 546.80, 541.20, 542.30, 71000000, 543.90),
('SPY',   '2024-11-01', 542.00, 548.50, 541.50, 547.20, 79000000, 545.10),
('TLT',   '2024-10-28', 92.80,  93.40,  91.90,  92.10,  18000000, 92.30),
('TLT',   '2024-10-29', 92.00,  92.80,  91.50,  92.40,  16000000, 92.10),
('TLT',   '2024-10-30', 92.50,  93.10,  91.80,  91.90,  14000000, 92.20),
('TLT',   '2024-10-31', 91.80,  92.20,  91.00,  91.40,  20000000, 91.60),
('TLT',   '2024-11-01', 91.50,  92.00,  90.80,  91.80,  17000000, 91.40),
('BTC',   '2024-10-28', 68500,  71200,  67800,  70900,  35000,    69800),
('BTC',   '2024-10-29', 70900,  73400,  70200,  72800,  42000,    71900),
('BTC',   '2024-10-30', 72500,  74100,  71800,  73200,  38000,    72700),
('BTC',   '2024-10-31', 73100,  75500,  72400,  74800,  51000,    74100),
('BTC',   '2024-11-01', 74800,  76200,  73900,  75600,  44000,    75100)
ON CONFLICT DO NOTHING;

INSERT INTO trade_market.order_book_snapshots (symbol, snapshot_ts, bid_price, ask_price, bid_size, ask_size) VALUES
('NVDA',  '2024-11-01 14:30:00+00', 154.85, 154.95, 12500, 9800),
('NVDA',  '2024-11-01 14:31:00+00', 154.90, 155.00, 11200, 10500),
('SPY',   '2024-11-01 14:30:00+00', 547.15, 547.20, 85000, 79000),
('TLT',   '2024-11-01 14:30:00+00',  91.78,  91.82, 42000, 38000),
('BTC',   '2024-11-01 14:30:00+00', 75580,  75620, 18,    22)
ON CONFLICT DO NOTHING;

INSERT INTO trade_market.signals (symbol, signal_date, signal_type, direction, confidence, description) VALUES
('NVDA',  '2024-11-01', 'momentum',      'long',    0.82,
 'NVDA breaking 5-day resistance at $151. Volume 28% above 20-day average. RSI at 65 — not yet overbought. Earnings beat catalyst still in play. Target $165 with stop at $148.'),
('SPY',   '2024-11-01', 'mean_reversion','neutral',  0.55,
 'SPY has rallied 2.6% in 5 sessions into historical resistance. Put/call ratio spiking to 1.42. VVIX elevated. Mean-reversion signal suggests consolidation before next directional move.'),
('TLT',   '2024-11-01', 'breakout',      'short',    0.74,
 'TLT breaking below $92 support on heavy volume (17M shares) suggesting continued rate pressure. 10-year yield pushing 4.65%. Risk-off positioning and Fed hawkish tone support further downside. Target $88.'),
('BTC',   '2024-11-01', 'momentum',      'long',    0.78,
 'BTC momentum accelerating through $75K psychological level. On-chain data: exchange outflows at 3-month high (bullish). Funding rates positive but not extreme. Halving cycle analysis points to further upside toward $82-85K.'),
('TSLA',  '2024-11-01', 'sentiment',     'long',    0.61,
 'Unusual options activity: 45,000 call contracts traded in TSLA vs 12,000 puts. Implied volatility skew shifted bullish. Analyst upgrades from two firms post Q3 delivery beat. Social sentiment score +0.71.')
ON CONFLICT DO NOTHING;
