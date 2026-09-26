"""Download or generate the dataset into data/ (gitignored).

Contract:
    - Idempotent. Re-running must not re-download.
    - Anything downloaded is checksum-verified.
    - Writes nothing outside data/.
"""
