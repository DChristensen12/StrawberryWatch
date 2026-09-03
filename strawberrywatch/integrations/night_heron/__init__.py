"""
Glue for Night Heron, the Django site and alert daemon at strawberrycreek.org.

Everything Night Heron needs from us lives under here. Their repository gets one
call, in email_alerts.py, and nothing else. If something about this integration
needs code, it goes in this package, not theirs.
"""

from strawberrywatch.integrations.night_heron.gnn_alerts import pending_alerts, reset_state

__all__ = ["pending_alerts", "reset_state"]
