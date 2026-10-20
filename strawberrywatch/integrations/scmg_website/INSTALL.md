# Mounting StrawberryWatch on strawberrycreek.org

The dashboard runs inside the site's existing process at `/strawberrywatch/`,
behind the site's own login. Visitors sign in on the same page Night Heron uses.

1. Install StrawberryWatch into the site's environment, without letting pip
   replace anything the site already has:

       pip install --no-deps <the StrawberryWatch commit>

   Any library StrawberryWatch adds goes in with `--only-binary=:all:`, since
   the server can't build newer packages from source.

2. Make StrawberryWatch's tables, with the database login in the site's `.env`:

       python -m strawberrywatch.integrations.scmg_website.setup_tables

   Run it again after every update. It adds any new tables and columns and
   leaves the rest alone.

3. At the bottom of `app/wsgi.py`, after `application = get_wsgi_application()`:

       from werkzeug.middleware.dispatcher import DispatcherMiddleware

       from strawberrywatch.integrations.scmg_website import MOUNT, make_dashboard

       application = DispatcherMiddleware(application, {MOUNT: make_dashboard()})

4. Optional, in `app/settings.py`, to let only one group change settings:

       STRAWBERRYWATCH_EDITOR_GROUP = "StrawberryWatch editors"

   Left unset, anyone who can sign in can edit. Superusers always can.

5. In the site's `.env`, turn on the dashboard's side of Night Heron's alerts:

       GNN_DASHBOARD=1

   Until this is set, `email_alerts` runs the models exactly as it did before,
   from `GNN_MODELS`. Once it is, the models and their settings come from the
   dashboard, and each check runs in a process of its own under the `website:`
   limits in StrawberryWatch's `settings.yaml`. StrawberryWatch emails its own
   list from that process, one email per person with an unsubscribe link.
   Night Heron only gets its alerts to text, when `GNN_ALERT_PHONES` is set.

6. Add StrawberryWatch's email account to the site's `.env`: the `ALERT_EMAIL_*`
   and `SMTP_*` lines from StrawberryWatch's `.env.example`. For now they hold
   the same account Night Heron sends from. The links in the emails start at
   `address` in the `website:` block of `settings.yaml`, so change it if the
   site moves.

7. `/strawberrywatch/signup` is open to everyone, along with the confirm and
   unsubscribe links it sends. It never shows anyone else's address. Link to it
   from the site's Alerts tab.

8. The models load from the folder `GNN_CHECKPOINT_DIR` in `.env` names, the
   same one Night Heron's hook uses. Settings > Model updates downloads newer
   model files from StrawberryWatch's main branch on GitHub into an `updates`
   folder inside it, so the site's user has to be able to write there.

9. Restart both `notifications_frontend` and `email_alerts`, then check the
   journal for a full model pass and Night Heron's own cycle lines:

       journalctl --user -u email_alerts

The dashboard's cookie is named `strawberrywatch` and limited to the
`/strawberrywatch` path, so it never meets Django's `sessionid` or `csrftoken`.
