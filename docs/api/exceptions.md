# Exception Hierarchy

All PrismLang exceptions inherit from `PrismLangError`. Catch the entire family
with `except PrismLangError` or target specific subtypes for granular handling.

```
PrismLangError
├── EncoderError
│   ├── ModelDownloadError
│   ├── ModelNotFoundError
│   └── TokenizerNotFoundError
├── TaxonomyError
│   ├── DuplicateCategoryError
│   ├── UnknownCategoryError
│   └── EmptyTaxonomyError
├── ProjectionError
│   ├── ZeroVectorError
│   └── DimensionMismatchError
├── CheckpointerError
│   ├── CheckpointerConnectionError
│   └── CheckpointerSchemaError
└── TenantError
    └── MissingTenantError
```

::: prismlang.exceptions
