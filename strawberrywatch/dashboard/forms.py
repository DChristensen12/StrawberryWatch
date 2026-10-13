"""
The forms, on Flask-WTF. They only validate: templates/macros.html draws each
field. Site and model choices are built per request from the inventory and the
serving registry, so a name that isn't one of theirs fails validation instead
of being stored.
"""

from __future__ import annotations

import math

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    PasswordField,
    RadioField,
    SelectMultipleField,
    StringField,
    TextAreaField,
)
from wtforms.validators import DataRequired, InputRequired, Length, Optional, ValidationError

from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import sources


def number_between(lo, hi, what):
    def check(_form, field):
        try:
            value = float(str(field.data).strip())
        except (TypeError, ValueError):
            raise ValidationError(f"{what} must be a number, like 1.0") from None
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValidationError(f"{what} must be between {lo:g} and {hi:g}")

    return check


class SetupForm(FlaskForm):
    source = RadioField(
        "Where should readings come from?",
        choices=[
            (sources.API, "The Strawberry Creek API. You'll need a token."),
            (sources.CSV, "A folder of CSV files on this computer"),
            (sources.REPLAY, "Try it on a past event first"),
        ],
        validators=[InputRequired(message="Choose where readings should come from")],
    )
    token = PasswordField("Token", validators=[Optional(), Length(max=400)])
    folder = StringField(
        "Folder",
        description="The full path of a folder with one CSV per site, named like south_fork_1.csv.",
        validators=[Optional(), Length(max=1000)],
    )
    event = RadioField("Which past event?", validators=[Optional()], choices=[])

    def __init__(self, *args, token_saved=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_saved = token_saved
        self.event.choices = [(e["folder"], e["title"]) for e in sources.replay_events()]

    def validate(self, extra_validators=None):
        """
        What each choice needs, checked after the fields themselves. Optional()
        stops a field's own checks when it's empty, so an empty token or folder
        has to be caught here or it would sail through.
        """
        ok = super().validate(extra_validators)
        kind = self.source.data
        if kind == sources.API and not (self.token.data or "").strip() and not self.token_saved:
            self.token.errors = [*self.token.errors, "Enter your API token"]
            ok = False
        if kind == sources.CSV:
            folder = (self.folder.data or "").strip()
            problem = "" if folder else "Enter the folder the CSV files are in"
            if folder:
                try:
                    sources.CsvSource(folder)
                except sources.SourceError as exc:
                    problem = str(exc)
            if problem:
                self.folder.errors = [*self.folder.errors, problem]
                ok = False
        if kind == sources.REPLAY and not self.event.data:
            self.event.errors = [*self.event.errors, "Choose an event to replay"]
            ok = False
        return ok


class TokenForm(FlaskForm):
    token = PasswordField(
        "New token",
        validators=[InputRequired(message="Enter your API token"), Length(max=400)],
    )


class NoteForm(FlaskForm):
    note = TextAreaField(
        "Add a note",
        description="Anything worth remembering, like what you found when you went to look.",
        filters=[lambda text: text.strip() if text else text],
        validators=[
            DataRequired(message="Write the note before adding it"),
            Length(max=2000, message="Keep the note under 2,000 characters"),
        ],
    )


def _strip(text):
    return text.strip() if text else text


class RecipientForm(FlaskForm):
    email = StringField(
        "Email address",
        filters=[_strip],
        validators=[DataRequired(message="Enter an email address"), Length(max=254)],
    )
    name = StringField(
        "Name",
        description="Optional. So you know who it is.",
        filters=[_strip],
        validators=[Optional(), Length(max=100, message="Keep the name under 100 characters")],
    )

    def validate_email(self, field):
        from strawberrywatch.dashboard import mail

        if not mail.valid(field.data):
            raise ValidationError("Enter a whole email address, with an @ and the part after it")


def settings_form(sites, labels, models):
    """Settings for everyone: which sites, how many alerts, rain, and which models are on."""

    class Form(FlaskForm):
        watch = SelectMultipleField(
            "Which sites to watch",
            choices=[(s, labels.get(s, s)) for s in sites],
            description="Unticked sites are still checked, but never flagged.",
            validators=[Optional()],
        )
        alerts = RadioField(
            "How many alerts do you want?",
            choices=[(key, f"{label}. {sentence}") for key, label, _, sentence in ws.PRESETS],
            validators=[Optional()],
        )
        rain = BooleanField(
            "Expect bigger swings during and after rain",
            description="Rain makes most readings look unusual. Leave this on unless you're testing something.",
        )

    for model in models:
        setattr(
            Form,
            f"model_{model.name}",
            RadioField(
                model.title,
                choices=[("on", "On"), ("off", "Off")],
                validators=[Optional()],
            ),
        )
    return Form


def advanced_form(models, support_options):
    """Advanced settings: each model's own values, the repeat window."""

    class Form(FlaskForm):
        repeat_hours = StringField(
            "Ignore repeats of the same thing for",
            validators=[number_between(*ws.REPEAT_RANGE, "Ignore repeats for")],
        )

    lo, hi = ws.SCALE_RANGE
    for model in models:
        setattr(
            Form,
            f"model_{model.name}",
            RadioField(
                model.title,
                choices=[("on", "On"), ("off", "Off")],
                validators=[Optional()],
            ),
        )
        for rule in model.rules:
            label = ws.RULE_WORDS.get(rule, (rule, ""))[0]
            setattr(
                Form,
                f"scale_{model.name}_{rule}",
                StringField(
                    label,
                    description=ws.RULE_WORDS.get(rule, ("", ""))[1],
                    validators=[number_between(lo, hi, f"{model.title}'s {label.lower()}")],
                ),
            )
        setattr(
            Form,
            f"rain_{model.name}",
            StringField(
                "Rain multiplier",
                description="How far the line rises while it's wet. 1 means no allowance for rain.",
                validators=[number_between(*ws.RAIN_RANGE, f"{model.title}'s rain multiplier")],
            ),
        )
        if "leave_out" in model.site_choice:
            setattr(
                Form,
                f"leave_{model.name}",
                SelectMultipleField(
                    "Leave a site out",
                    choices=[(s, s) for s in model.sites],
                    description=(
                        "A site left out is treated as missing before the model reads it, so it "
                        "neither feeds its neighbors nor gets checked."
                    ),
                    validators=[Optional()],
                ),
            )
        setattr(
            Form,
            f"support_{model.name}",
            SelectMultipleField(
                "Support modules",
                choices=[(o["name"], o["title"]) for o in support_options[model.name]],
                validators=[Optional()],
            ),
        )
    return Form
