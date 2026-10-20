"""
What integrations/scmg_website may touch, enforced. It gets the website's
database login, a username and whether that person may edit. That's all.
A reviewer only has to check this test is still here.
"""

import re
from pathlib import Path

SHELL = Path(__file__).resolve().parent.parent / "strawberrywatch" / "integrations" / "scmg_website"

FORBIDDEN = (
    "password",
    "auth_user",
    "get_user_model",
    "User.objects",
    "authenticate",
    "check_password",
    "make_password",
    "SECRET_KEY",
    "CLOUDFLARE",
    # The database login can reach Django's own tables. Their prefixes are out too
    "auth_",
    "django_session",
)
# The one exception: the database setting it shares with Night Heron
ALLOWED = ("MYSQL_DATABASE_PASSWORD",)


def _files():
    files = [p for p in SHELL.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert files, f"nothing found in {SHELL}"
    return files


def test_the_website_shell_reaches_for_nothing_it_shouldnt():
    for path in _files():
        text = path.read_text()
        for allowed in ALLOWED:
            text = text.replace(allowed, "")
        for word in FORBIDDEN:
            assert word.lower() not in text.lower(), f"{path.name} mentions {word!r}"


def test_it_never_logs_who_someone_is():
    logging_call = re.compile(r"logger\.\w+\((.*)\)")
    for path in _files():
        if path.suffix != ".py":
            continue
        for call in logging_call.findall(path.read_text()):
            for word in ("user", "session", "cookie", "header", "environ"):
                assert word not in call.lower(), f"{path.name} logs {call!r}"


def test_it_only_reads_its_own_settings():
    for path in _files():
        if path.suffix != ".py":
            continue
        text = path.read_text()
        # os.environ and getenv are .env; a WSGI environ's keys are the request's, not settings
        for name in re.findall(r"getenv\(\s*['\"]([A-Z_]+)", text) + re.findall(
            r"os\.environ(?:\.get\(|\[)\s*['\"]([A-Z_]+)", text
        ):
            assert name.startswith(("MYSQL_", "STRAWBERRYWATCH_")), f"{path.name} reads {name}"
