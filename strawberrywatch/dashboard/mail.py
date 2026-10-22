"""
Emails the list when a check flags something new. Locally both the account and
the list live in .env. Everyone gets their own copy. Nobody sees who else is
on the list, and one bad address can't stop the rest.

On the website the list is in the database, and the check's own process sends
an HTML email to each person with an unsubscribe link. Replays don't send.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import smtplib
import ssl
import tempfile
import time
from email.message import EmailMessage
from email.utils import formataddr, getaddresses
from pathlib import Path

from dotenv import dotenv_values, find_dotenv, load_dotenv

from strawberrywatch.dashboard import view
from strawberrywatch.watch import sources

logger = logging.getLogger(__name__)

ACCOUNT = ("ALERT_EMAIL_SENDER", "ALERT_EMAIL_PASSWORD", "SMTP_SERVER", "SMTP_PORT")
LIST_KEY = "ALERT_EMAIL_RECIPIENTS"
# Strict enough that an address can't sneak in a header, a second address, or
# a quote that ends the .env line early
ADDRESS = re.compile(r"^[^@\s,;<>\"'\\]+@[^@\s,;<>\"'\\]+\.[^@\s,;<>\"'\\.]+$")
# Names go on the same header line and follow the same rules
NAME = re.compile(r"^[^,;<>\"'\\\r\n]*$")
# Long enough for a slow resolver, short enough that saving doesn't hang
LOOKUP_SECONDS = 5
FOOTER = (
    "Sent by StrawberryWatch because you're on its email list. To stop getting "
    "these, ask whoever runs StrawberryWatch to take you off it."
)


class MailError(RuntimeError):
    """An email or address problem, worded so it's safe to show on the page."""


def valid(address):
    return len(address) <= 254 and bool(ADDRESS.match(address))


def check_address(address):
    """
    Returns (address to save, note). Checks the format and that the domain
    takes mail, which catches typos like "berkely.edu". If the lookup can't
    run, the address is saved anyway and the note says so.
    """
    from email_validator import EmailNotValidError, EmailUndeliverableError, validate_email

    address = (address or "").strip()
    if "@" not in address:
        raise MailError("Enter an email address with an @")
    if not valid(address):
        raise MailError("Enter an email address in the correct format")
    domain = address.rsplit("@", 1)[1]
    try:
        info = validate_email(address, check_deliverability=True, timeout=LOOKUP_SECONDS)
    except EmailUndeliverableError as exc:
        text = str(exc)
        if text.startswith("There was an error while checking"):
            return (
                address,
                f"Couldn't confirm that {domain} accepts email. {address} was added anyway.",
            )
        if "does not exist" in text:
            raise MailError(f"{domain} doesn't exist. Check the spelling after the @") from None
        raise MailError(f"{domain} doesn't accept email. Check the spelling after the @") from None
    except EmailNotValidError as exc:
        raise MailError(f"Enter an email address in the correct format. {exc}") from None
    normalized = info.normalized
    if getattr(info, "mx", None) is None:
        # The lookup timed out or found no nameservers. That tells us little about the address
        return (
            normalized,
            f"Couldn't confirm that {domain} accepts email. {normalized} was added anyway.",
        )
    return normalized, ""


def check_name(name):
    if name and not NAME.match(name):
        raise MailError("Name can't include quotes, commas, semicolons or angle brackets")
    return (name or "").strip()


# The list


def env_path():
    """The .env the alerts read, or a new one beside where StrawberryWatch was started."""
    found = find_dotenv(usecwd=True)
    return Path(found) if found else Path.cwd() / ".env"


def _parse(line):
    return [
        {"email": address, "name": name or None}
        for name, address in getaddresses([line or ""])
        if address
    ]


class EnvList:
    """The local list, one line of .env."""

    def __init__(self, path=None):
        self.path = Path(path) if path else env_path()

    def people(self):
        if not self.path.exists():
            return []
        return _parse(dotenv_values(self.path).get(LIST_KEY))

    def add(self, email, name=None):
        """Returns False if the address is already on the list, ignoring case."""
        people = self.people()
        if any(p["email"].lower() == email.lower() for p in people):
            return False
        self._write([*people, {"email": email, "name": name or None}])
        return True

    def remove(self, email):
        people = self.people()
        kept = [p for p in people if p["email"].lower() != email.lower()]
        if len(kept) == len(people):
            return False
        self._write(kept)
        return True

    def _write(self, people):
        """
        Rewrites only the ALERT_EMAIL_RECIPIENTS line. Writes a temp file and
        renames it over .env, and a crash can't leave half a file. The old one is
        kept as .env.bak, and both are owner-only since .env has the email password.
        """
        value = ", ".join(formataddr((p["name"] or "", p["email"])) for p in people)
        line = f"{LIST_KEY}='{value}'"
        old = self.path.read_text() if self.path.exists() else ""
        lines = old.splitlines(keepends=True)
        pattern = re.compile(rf"^\s*(export\s+)?{LIST_KEY}\s*=")
        ending = "\r\n" if old.endswith("\r\n") else "\n"
        found = False
        for i, existing in enumerate(lines):
            if pattern.match(existing):
                newline = existing[len(existing.rstrip("\r\n")) :] or ending
                lines[i] = line + newline
                found = True
                break
        if not found:
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += ending
            lines.append(line + ending)
        if self.path.exists():
            backup = self.path.with_name(self.path.name + ".bak")
            shutil.copyfile(self.path, backup)
            sources_private(backup)
        handle, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".env.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as f:
                f.write("".join(lines))
            sources_private(temp)
            os.replace(temp, self.path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
        sources_private(self.path)


class StoreList:
    """The website's list, a database table StrawberryWatch sends to."""

    def __init__(self, store):
        self.store = store

    def people(self):
        return self.store.recipients()

    def add(self, email, name=None, label_id=None, added_by=None):
        return self.store.add_recipient(email, name, label_id, added_by)

    def remove(self, email):
        for person in self.store.recipients():
            if person["email"].lower() == email.lower():
                self.store.remove_recipient(person["id"])
                return True
        return False


def sources_private(path):
    from strawberrywatch.watch.store import private

    private(path)


def mailing_list(store, in_env=True):
    """
    The list for this profile. Locally that's .env, and anyone left on the old
    SQLite list gets moved over.
    """
    if not in_env:
        return StoreList(store)
    book = EnvList()
    leftover = store.recipients()
    if leftover:
        for person in leftover:
            book.add(person["email"], person["name"])
        store.clear_recipients()
    return book


# Sending


def missing():
    """The account settings missing from .env. Empty means we can send."""
    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv()
    return [name for name in ACCOUNT if not os.getenv(name)]


def sender():
    """The From address. ALERT_EMAIL_FROM when the account logs in under a different name."""
    missing()
    return os.getenv("ALERT_EMAIL_FROM") or os.getenv("ALERT_EMAIL_SENDER") or ""


def is_on(store):
    return store.pref("email_on") == "yes"


def send(subject, body, people):
    """
    Sends one plain copy to each person over a single connection. Returns
    [(email, ok, message)], or raises MailError if no email went out.
    """

    def plain(address, person):
        message = EmailMessage()
        message["From"] = address
        message["To"] = person["email"]
        message["Subject"] = subject
        message.set_content(body)
        return message

    return deliver(people, plain)


def deliver(people, build):
    """
    Sends build(sender address, person) to each person over one connection.
    Returns [(email, ok, message)], or raises MailError if no email went out.
    """
    gaps = missing()
    if gaps:
        raise MailError(f"Email isn't set up on this computer. .env needs {', '.join(gaps)}.")
    if not people:
        raise MailError("There's nobody on the list.")
    account = os.getenv("ALERT_EMAIL_SENDER")
    address = sender()
    try:
        port = int(os.getenv("SMTP_PORT"))
    except ValueError:
        raise MailError("SMTP_PORT in .env isn't a number.") from None
    context = ssl.create_default_context()
    results = []
    try:
        # 465 is TLS from the start; other ports upgrade with STARTTLS
        if port == 465:
            smtp = smtplib.SMTP_SSL(os.getenv("SMTP_SERVER"), port, timeout=30, context=context)
        else:
            smtp = smtplib.SMTP(os.getenv("SMTP_SERVER"), port, timeout=30)
            smtp.starttls(context=context)
        with smtp:
            smtp.login(account, os.getenv("ALERT_EMAIL_PASSWORD"))
            for person in people:
                message = build(address, person)
                try:
                    smtp.send_message(message, from_addr=address, to_addrs=[person["email"]])
                    results.append((person["email"], True, "Sent"))
                except smtplib.SMTPRecipientsRefused:
                    results.append(
                        (person["email"], False, "The mail server turned this address down.")
                    )
                except smtplib.SMTPServerDisconnected:
                    raise
                except smtplib.SMTPException as exc:
                    results.append((person["email"], False, f"Didn't go: {exc}"))
    except smtplib.SMTPAuthenticationError:
        raise MailError("The email account in .env turned down the password.") from None
    except (smtplib.SMTPException, OSError) as exc:
        if not results:
            raise MailError(f"The email didn't go: {exc}") from None
        done = {email for email, _, _ in results}
        results += [
            (p["email"], False, "The mail server hung up before this one went.")
            for p in people
            if p["email"] not in done
        ]
    return results


def _record(store, kind, people, results, subject, check_id=None):
    for email, ok, message in results:
        store.record_email_result(email, kind, ok, message)
    sent = sum(1 for _, ok, _ in results if ok)
    if sent == len(people):
        store.log_email(kind, sent, True, subject, check_id)
    else:
        store.log_email(
            kind, sent, False, f"{len(people) - sent} of {len(people)} didn't go.", check_id
        )
    return sent


def alert(store, inventory, items):
    """Subject and plain-text body for one check's alerts, for the local list."""
    views = [view.item_view(store, item, None) for item in items]
    subject, opener, blocks = alert_parts(views, inventory)
    return subject, as_text(opener, blocks, FOOTER)


def alert_parts(views, inventory):
    """Subject, opening line and one block per alert, worded like the pages."""
    if len(views) == 1:
        subject = f"StrawberryWatch: {views[0]['label']} alert at {views[0]['site_name']}"
        opener = "New alert in Strawberry Creek."
    else:
        subject = f"StrawberryWatch: {len(views)} alerts"
        opener = f"{len(views)} new alerts in Strawberry Creek."
    blocks = []
    for v in views:
        # The full name, landmark and all, since the reader may not know the site codes
        place = inventory.site(v["site"]).name if v["site"] in inventory.sites else v["site_name"]
        blocks.append(
            {
                "heading": f"{v['label'].capitalize()} alert at {place}",
                "sentence": f"{v['happened']} Flagged by {v['title']}.",
                "extras": list(v["extras"]),
            }
        )
    return subject, opener, blocks


def as_text(opener, blocks, footer):
    parts = [opener, ""]
    for block in blocks:
        parts += [block["heading"], block["sentence"], *block["extras"], ""]
    parts.append(footer)
    return "\n".join(parts)


def after_check(store, inventory, check_id):
    """
    Emails the list about new alerts from this check. Doesn't raise. A mail
    problem can't stop the checks. Failures are logged and shown on the Email page.
    """
    try:
        if check_id is None or not is_on(store):
            return
        if not sources.is_live((store.source() or {}).get("kind")):
            return
        items = [n for n in store.noticed(flagged=True) if n["check_id"] == check_id]
        people = mailing_list(store).people()
        if not items or not people:
            return
        subject, body = alert(store, inventory, items)
        try:
            results = send(subject, body, people)
        except MailError as exc:
            for person in people:
                store.record_email_result(person["email"], "alert", False, str(exc))
            store.log_email("alert", 0, False, str(exc), check_id)
            logger.warning("alert email didn't go: %s", exc)
            return
        _record(store, "alert", people, results, subject, check_id)
    except Exception:
        logger.exception("emailing after check %s failed", check_id)


def chosen(people, only):
    """The people whose address is in only, ignoring case. None means everyone."""
    if only is None:
        return people
    wanted = {address.lower() for address in only}
    return [p for p in people if p["email"].lower() in wanted]


def test(store, only=None):
    """Sends a test email to everyone on the list, or the addresses in only. Raises MailError."""
    people = chosen(mailing_list(store).people(), only)
    body = (
        "This is a test from StrawberryWatch. If you got it, you're on its email "
        "list, and alerts about unusual readings in Strawberry Creek will reach you.\n\n" + FOOTER
    )
    try:
        results = send("StrawberryWatch test email", body, people)
    except MailError as exc:
        for person in people:
            store.record_email_result(person["email"], "test", False, str(exc))
        store.log_email("test", 0, False, str(exc))
        raise
    return _record(store, "test", people, results, "StrawberryWatch test email"), len(people)


# The website's own emails. Night Heron doesn't email for StrawberryWatch there.

SENDER_NAME = "StrawberryWatch"
WEBSITE_FOOTER = "You're getting this because you're on StrawberryWatch's alert list."
# Inline images. Mail apps show them without fetching from the web
LOGOS = {"emblem": "email/emblem.png", "sponsors": "email/sponsors.png"}
INLINE = {name: f"cid:{name}" for name in LOGOS}
_templates = None


def render(name, **context):
    """An email's HTML from the dashboard's templates, outside any request."""
    global _templates
    if _templates is None:
        from jinja2 import Environment, PackageLoader

        _templates = Environment(loader=PackageLoader("strawberrywatch.dashboard"), autoescape=True)
    return _templates.get_template(name).render(**context)


def unsubscribe_link(address, token):
    return f"{address}/unsubscribe/{token}"


def footer_text(link):
    """The plain-text footer. The page it links to also offers a week's pause."""
    return f"{WEBSITE_FOOTER} To pause them for a week or unsubscribe: {link}"


def models_for(person):
    """The models one person gets alerts from, or None for every model, new ones too."""
    return None if person.get("models") is None else json.loads(person["models"])


def paused(person, now):
    """A pause with no end lasts until someone turns it off. Past ones just lapse."""
    if person.get("paused_at") is None:
        return False
    return person.get("paused_until") is None or now < person["paused_until"]


def picked(person, items, now):
    """The positions in items of the alerts this person gets, none while they're paused."""
    if paused(person, now):
        return []
    wanted = models_for(person)
    return [i for i, item in enumerate(items) if wanted is None or item["model"] in wanted]


def html_message(address, to, subject, text, html, unsubscribe=None):
    """One email to one person: plain text, an HTML version, and the logos inside it."""
    message = EmailMessage()
    message["From"] = formataddr((SENDER_NAME, address))
    message["To"] = to
    message["Subject"] = subject
    if unsubscribe:
        message["List-Unsubscribe"] = f"<{unsubscribe}>"
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    # Same layout as the inline image example in Python's email docs
    html_part = message.get_payload()[1]
    folder = Path(__file__).resolve().parent / "static"
    for name, path in LOGOS.items():
        html_part.add_related(
            (folder / path).read_bytes(), maintype="image", subtype="png", cid=f"<{name}>"
        )
    return message


def alert_email(subject, opener, blocks, unsubscribe, images=INLINE):
    """(text, html) for one person's alert email."""
    text = as_text(opener, blocks, footer_text(unsubscribe))
    html = render(
        "email_message.html",
        subject=subject,
        opener=opener,
        blocks=blocks,
        footer=WEBSITE_FOOTER,
        unsubscribe=unsubscribe,
        images=images,
    )
    return text, html


def website_alert(store, inventory, check_id, address, items=None):
    """
    Emails the website's list about new alerts from this check, or just the
    items given, one email per person with their own unsubscribe link. Each
    person only hears about the models they picked, and nothing while paused.
    Doesn't raise, same as after_check.
    """
    try:
        if check_id is None or not is_on(store):
            return
        if items is None:
            items = [n for n in store.noticed(flagged=True) if n["check_id"] == check_id]
        everyone = store.recipients()
        chosen = {p["id"]: picked(p, items, time.time()) for p in everyone}
        people = [p for p in everyone if chosen[p["id"]]]
        if not items or not people:
            return
        views = [view.item_view(store, item, None) for item in items]
        # The log keeps one subject per check, the one for every alert in it
        subject, _, _ = alert_parts(views, inventory)

        def build(sender_address, person):
            mine, opener, blocks = alert_parts([views[i] for i in chosen[person["id"]]], inventory)
            link = unsubscribe_link(address, person["unsubscribe_token"])
            text, html = alert_email(mine, opener, blocks, link)
            return html_message(sender_address, person["email"], mine, text, html, link)

        try:
            results = deliver(people, build)
        except MailError as exc:
            for person in people:
                store.record_email_result(person["email"], "alert", False, str(exc))
            store.log_email("alert", 0, False, str(exc), check_id)
            logger.warning("alert email didn't go: %s", exc)
            return
        _record(store, "alert", people, results, subject, check_id)
    except Exception:
        logger.exception("emailing after check %s failed", check_id)


def website_test(store, address, only=None):
    """A test email to everyone on the website's list, or the addresses in only. Raises MailError."""
    people = chosen(store.recipients(), only)
    subject = "StrawberryWatch test email"
    opener = (
        "This is a test from StrawberryWatch. If you got it, alerts about Strawberry Creek "
        "will reach you too."
    )

    def build(sender_address, person):
        link = unsubscribe_link(address, person["unsubscribe_token"])
        text, html = alert_email(subject, opener, [], link)
        return html_message(sender_address, person["email"], subject, text, html, link)

    try:
        results = deliver(people, build)
    except MailError as exc:
        for person in people:
            store.record_email_result(person["email"], "test", False, str(exc))
        store.log_email("test", 0, False, str(exc))
        raise
    return _record(store, "test", people, results, subject), len(people)
