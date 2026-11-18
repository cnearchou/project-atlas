"""Data ingestion clients (Technical Reference, Part 2).

Each module wraps one source listed in Part 2 of the Technical Reference. Clients only download
data and save it in the RAW layer (Spec §0.1); cleaning and point-in-time handling happen in
atlas.data. API keys come from the .env file (see .env.example) and are never hard-coded.

The `synthetic` module generates realistic fake data with the same table layouts, so the full
pipeline and the tests can run without any licensed data or API keys.
"""
