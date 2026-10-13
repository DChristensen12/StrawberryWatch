"""
Emailing the people on StrawberryWatch's list when a check notices something
new. It sends through the account in .env that main.py's alerts use, and every
address goes on the envelope only, so nobody sees who else is on the list.

Replays never send: they're past events, and nobody should be told about a
2025 spill as if it were happening now.
"""

from __future__ import annotations

import logging
import os
import re
import smtplib
import ssl
from email.message import EmailMessage

from dotenv import find_dotenv, load_dotenv

from strawberrywatch.dashboard import view
from strawberrywatch.watch import sources

logger = logging.getLogger(__name__)

ACCOUNT = ("ALERT_EMAIL_SENDER", "ALERT_EMAIL_PASSWORD", "SMTP_SERVER", "SMTP_PORT")
# Plain enough to catch typos, strict enough that no address can carry a header or a second address
ADDRESS = re.compile(r"^[^@\s,;<>\"']+@[^@\s,;<>\"']+\.[^@\s,;<>\"'.]+$")
FOOTER = (
    "Sent by StrawberryWatch because you're on its email list. To stop getting "
    "these, ask whoever runs StrawberryWatch to take you off it."
)


class MailError(RuntimeError):
    """Why an email didn't go, in words that are safe to show on the page."""


def valid(address):
    return len(address) <= 254 and bool(ADDRESS.match(address))


def missing():
    """Which of the account's settings .env doesn't have. Empty when sending can work."""
    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv()
    return [name for name in ACCOUNT if not os.getenv(name)]


def sender():
    missing()
    return os.getenv("ALERT_EMAIL_SENDER") or ""


def is_on(store):
    return store.pref("email_on") == "yes"


def send(subject, body, people):
    """Send one email to everyone in people. Raises MailError."""
    gaps = missing()
    if gaps:
        raise MailError(f"Email isn't set up on this computer. .env needs {', '.join(gaps)}.")
    if not people:
        raise MailError("There's nobody on the list.")
    address = os.getenv("ALERT_EMAIL_SENDER")
    try:
        port = int(os.getenv("SMTP_PORT"))
    except ValueError:
        raise MailError("SMTP_PORT in .env isn't a number.") from None
    message = EmailMessage()
    message["From"] = address
    message["To"] = "undisclosed-recipients:;"
    message["Subject"] = subject
    message.set_content(body)
    context = ssl.create_default_context()
    try:
        # 465 is TLS from the start; anything else upgrades with STARTTLS
        if port == 465:
            smtp = smtplib.SMTP_SSL(os.getenv("SMTP_SERVER"), port, timeout=30, context=context)
        else:
            smtp = smtplib.SMTP(os.getenv("SMTP_SERVER"), port, timeout=30)
            smtp.starttls(context=context)
        with smtp:
            smtp.login(address, os.getenv("ALERT_EMAIL_PASSWORD"))
            smtp.send_message(message, from_addr=address, to_addrs=[p["email"] for p in people])
    except smtplib.SMTPAuthenticationError:
        raise MailError("The email account in .env turned down the password.") from None
    except (smtplib.SMTPException, OSError) as exc:
        raise MailError(f"The email didn't go: {exc}") from None


def alert(store, inventory, items):
    """The subject and body for what one check noticed, in the same sentences as the pages."""
    views = [view.item_view(store, item, None) for item in items]
    if len(views) == 1:
        subject = f"StrawberryWatch: unusual {views[0]['label']} at {views[0]['site']}"
    else:
        subject = f"StrawberryWatch: {len(views)} unusual readings"
    parts = ["StrawberryWatch noticed something unusual in Strawberry Creek.", ""]
    for v in views:
        name = inventory.site(v["site"]).name if v["site"] in inventory.sites else ""
        parts.append(f"Unusual {v['label']} at {v['site']}" + (f" ({name})" if name else ""))
        parts.append(f"{v['happened']} {v['title']} noticed.")
        parts.extend(v["extras"])
        parts.append("")
    parts.append(FOOTER)
    return subject, "\n".join(parts)


def after_check(store, inventory, check_id):
    """
    Email the list about anything a check newly flagged. Never raises, so a
    mail problem can't stop the checks; it's logged and shown on Email alerts.
    """
    try:
        if check_id is None or not is_on(store):
            return
        if (store.source() or {}).get("kind") == sources.REPLAY:
            return
        items = [n for n in store.noticed(flagged=True) if n["check_id"] == check_id]
        people = store.recipients()
        if not items or not people:
            return
        subject, body = alert(store, inventory, items)
        try:
            send(subject, body, people)
        except MailError as exc:
            store.log_email("alert", len(people), False, str(exc), check_id)
            logger.warning("alert email didn't go: %s", exc)
            return
        store.log_email("alert", len(people), True, subject, check_id)
    except Exception:
        logger.exception("emailing after check %s failed", check_id)


def test(store):
    """A short email to everyone on the list, to check it reaches them. Raises MailError."""
    people = store.recipients()
    body = (
        "This is a test from StrawberryWatch. If you got it, you're on its email "
        "list, and alerts about unusual readings in Strawberry Creek will reach you.\n\n" + FOOTER
    )
    try:
        send("StrawberryWatch test email", body, people)
    except MailError as exc:
        store.log_email("test", len(people), False, str(exc))
        raise
    store.log_email("test", len(people), True, "StrawberryWatch test email")
