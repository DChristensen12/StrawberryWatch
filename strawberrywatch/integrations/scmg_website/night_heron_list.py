"""
The addresses on Night Heron's alert rules, for copying onto StrawberryWatch's
list once. After that the two lists are separate.
"""

from __future__ import annotations

import re

# Night Heron's three kinds of alert rule, each with a comma-separated email list
RULE_TABLES = ("base_staticthreshold", "base_movingthreshold", "base_advancedalert")
# Night Heron's forms start the field with made-up addresses until someone types real ones
PLACEHOLDER = re.compile(r"^example\d*@", re.IGNORECASE)


def addresses():
    """Every address on an active Night Heron rule, once each, in the order first seen."""
    from strawberrywatch.ingest import sql_client

    seen = {}
    with sql_client._connect() as conn:
        cursor = conn.cursor()
        for table in RULE_TABLES:
            try:
                # The table names are the constants above
                cursor.execute(f"SELECT email_list FROM `{table}` WHERE active = 1")  # nosec B608
            except Exception:
                continue  # a site without this kind of rule
            for (field,) in cursor.fetchall():
                for part in (field or "").replace(";", ",").split(","):
                    address = part.strip()
                    if address and not PLACEHOLDER.match(address):
                        seen.setdefault(address.lower(), address)
    return list(seen.values())
