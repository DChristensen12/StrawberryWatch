"""
What differs between the local app and the website. The pages are the same,
only sign-in, email and who runs the checks change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    # The list is a line in .env and we send the emails. Otherwise it's in the
    # database and Night Heron sends them.
    email_in_env: bool
    # Sign in with the one-time link printed at startup, and only answer on this computer
    token_login: bool
    # Checks run in a thread here. On the website Night Heron runs each one in
    # its own process, with a time limit.
    checks_here: bool
    # Each model's minutes between alerts, in Advanced settings
    alert_gaps: bool = False


LOCAL = Profile(name="local", email_in_env=True, token_login=True, checks_here=True)

WEBSITE = Profile(
    name="website", email_in_env=False, token_login=False, checks_here=False, alert_gaps=True
)
