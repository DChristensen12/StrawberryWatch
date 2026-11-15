# Mounting StrawberryWatch on strawberrycreek.org

Here, we describe how the StrawberryWatch dashboard runs inside the website's existing process at `/strawberrywatch/`, behind the site's own login. Visitors sign in on the same page Night Heron uses. The sections below follow the order in which we install it.

## Installing the package

We install StrawberryWatch into the site's environment without letting pip replace any library the site already has:

    pip install --no-deps <the StrawberryWatch commit>

Every library StrawberryWatch adds goes in with `--only-binary=:all:`, since the server cannot build newer packages from source. The dashboard needs the libraries below, which make up the `dashboard` extra in `pyproject.toml`:

    pip install --only-binary=:all: flask flask-wtf wtforms email-validator platformdirs

## Creating the tables

StrawberryWatch keeps its own data in sw_ tables on the website's database, created with the database login in the site's `.env`. The command reads the `.env` in the folder it is run from, and we run it from the site's folder:

    cd ~/notifications_frontend
    python -m strawberrywatch.integrations.scmg_website.setup_tables

We recommend running it again after every update. It adds any new tables and columns and leaves the existing ones as they are.

## Mounting the dashboard

At the bottom of `app/wsgi.py`, after `application = get_wsgi_application()`, we add:

    from werkzeug.middleware.dispatcher import DispatcherMiddleware

    from strawberrywatch.integrations.scmg_website import MOUNT, make_dashboard

    application = DispatcherMiddleware(application, {MOUNT: make_dashboard()})

## Limiting who can edit

This step is optional. One line in `app/settings.py` lets only one group change settings:

    STRAWBERRYWATCH_EDITOR_GROUP = "StrawberryWatch editors"

When the setting is left out, anyone who can sign in can edit. Superusers can always edit.

## Turning on the dashboard's alerts

One line in the site's `.env` turns on the dashboard's side of Night Heron's alerts:

    GNN_DASHBOARD=1

Until this line is set, `email_alerts` runs the models exactly as before, from `GNN_MODELS`. Once it is set, the models and their settings come from the dashboard, and each check runs in a process of its own under the `website:` limits in StrawberryWatch's `settings.yaml`. StrawberryWatch emails its own list from that process, one email per person with an unsubscribe link. Night Heron's only part in these alerts is sending them as text messages, when `GNN_ALERT_PHONES` is set.

## Email

StrawberryWatch sends email through Night Heron's email API account, using the account id, API token and `FROM_EMAIL` lines that Night Heron already reads from the site's `.env`. Nothing needs to be added. The `ALERT_EMAIL_*` and `SMTP_*` lines from older installs are no longer read and can be removed. Links in the emails start at `address` in the `website:` block of `settings.yaml`, which we change if the site moves.

## Recipients and pauses

The unsubscribe links in the alert emails are open to everyone, and the pages behind them do not show anyone else's address. There is no public sign-up, since Environment, Health & Safety (EH&S) asked that only people added on Settings > Email alerts receive the alerts. Each person's name on that page opens their alert settings: (1) which models they receive alerts from, and (2) a pause of 1 day, 1 week, 30 days, until a date, or until turned back on. The unsubscribe page also lets people pause their own alerts for 1 week.

## Model files

The models load from the folder that `GNN_CHECKPOINT_DIR` in `.env` names, the same folder Night Heron's hook uses. Inside it, each model has a folder of its own, named after the model, following the layout of `checkpoints/` in the repository. Files that an older install left loose in the folder still load, and we recommend moving them into their model's folder when convenient. Settings > Model updates downloads newer model files from StrawberryWatch's main branch on GitHub into an `updates` folder inside it. Thus, the site's user needs to be able to write there.

## Restarting

Finally, we restart both `notifications_frontend` and `email_alerts`, then check the journal for a full model pass and Night Heron's own cycle lines:

    journalctl --user -u email_alerts

The dashboard's cookie is named `strawberrywatch` and is limited to the `/strawberrywatch` path, which keeps it apart from Django's `sessionid` and `csrftoken` cookies.

## Moving between the two dashboards

One sign-in covers both dashboards. The site's Alerts link therefore goes to `/alerts/`, which sends people to the dashboard they were last on and stores that choice in the site's session. Signing in on a card sets the choice, StrawberryWatch's own pages set it as they are read, and the Night Heron link in StrawberryWatch's navigation bar (and the StrawberryWatch link in Night Heron's) sets it the other way. Signed out, `/alerts/` goes to the login page with both cards.
