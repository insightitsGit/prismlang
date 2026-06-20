"""Finance-domain taxonomy for the PrismLang demo.

Three categories that cover typical financial multi-agent workflows:
  risk        — credit risk, market risk, exposure, volatility
  market      — pricing, equities, FX, commodities, indices
  compliance  — regulatory, AML, KYC, reporting, audit
"""

from prismlang import Category, TaxonomyConfig

FINANCE_TAXONOMY = TaxonomyConfig(
    categories=[
        Category(
            slug="risk",
            label="Risk Management",
            keywords=[
                "risk", "exposure", "volatility", "credit", "default",
                "hedging", "drawdown", "var", "stress", "scenario",
                "counterparty", "collateral", "liquidity", "loss",
            ],
        ),
        Category(
            slug="market",
            label="Market Analysis",
            keywords=[
                "market", "price", "equity", "stock", "bond", "yield",
                "fx", "currency", "commodity", "index", "return", "asset",
                "portfolio", "valuation", "rate", "spread", "trade",
            ],
        ),
        Category(
            slug="compliance",
            label="Regulatory Compliance",
            keywords=[
                "compliance", "regulation", "regulatory", "aml", "kyc",
                "reporting", "audit", "law", "requirement", "policy",
                "disclosure", "fiduciary", "sanction", "sec", "finra",
            ],
        ),
    ],
    alpha=0.3,
)
