# data/

Local storage layers of Spec §0.1. Everything in `raw/`, `clean/`, `features/` and `outputs/` is
ignored by git, because licensed data must never be published and outputs are reproducible.

| Folder | Contents |
|---|---|
| `raw/` | Downloads exactly as received; write-once (a new download gets a new, dated table name) |
| `clean/` | Adjusted, point-in-time, survivorship-free tables |
| `features/` | One value per company per date per signal |
| `outputs/` | Forecasts, risk, portfolio, research registry and monitoring tables |
