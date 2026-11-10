"""Local history shell. All imported text is displayed through textContent."""


def render_history():
    from .render import _document

    return _document(
        title="Private report history",
        page_class="history-page",
        body="""
<main class="history-shell"><header><p class="report-kicker">On this Mac · encrypted at rest</p>
<h1>Your private report history</h1><p>Save only when you choose. Ending a temporary session leaves saved history in place.</p>
<a href="/">New temporary analysis</a></header>
<p data-history-status role="status">Opening history…</p>
<section><h2>Saved reports</h2><p>100 reports maximum · 6 MB per report · 200 MB total. Report-only saves omit photos and point geometry.</p>
<button type="button" data-history-refresh>Refresh history</button><div data-history-list></div>
<button type="button" data-history-compare>Compare two selected reports</button></section>
<section><h2>Restore a password-protected backup</h2>
<p>A backup password cannot be recovered. Losing the Mac's Keychain key makes local archives unreadable unless you kept a password-protected backup.</p>
<form method="post" action="/history/locked" data-history-import><fieldset disabled data-history-controls><label>Backup file<input type="file" name="backup" accept=".fab" required></label>
<label>Backup password<input type="password" name="password" minlength="12" maxlength="1024" autocomplete="off" required></label>
<button>Import encrypted backup</button></fieldset></form></section>
<section data-history-detail aria-live="polite"></section>
<p>Deleting removes this app's copy. SSD recovery, OS backups and independently exported files are outside this deletion. Imported measurements are unverified observations. This app does not identify people from photographs.</p></main>""",
    )


def session_panel(session_id, csrf):
    import html

    def esc(s):
        return html.escape(str(s), quote=True)

    return f'''<section class="archive-panel" data-history-session="{esc(session_id)}" data-history-session-csrf="{esc(csrf)}">
<h2>Keep this report?</h2><p>This session is temporary. Saving creates a separate encrypted numerical snapshot. Point geometry and photos are kept only for photos you explicitly select.</p>
<p data-history-status role="status">Open the app to enable private history.</p>
<form method="post" action="/history/locked" data-history-save><fieldset disabled data-history-controls><label>Local profile name<input name="profile" maxlength="120" required placeholder="Choose a name; no face matching"></label>
<label>Report name<input name="name" maxlength="120" placeholder="Optional label"></label>
<label>Capture conditions<select name="protocol"><option value="unspecified">Not recorded</option><option value="casual">Casual capture</option><option value="controlled">Controlled repeat setup</option><option value="mixed">Mixed conditions</option></select></label>
<label>Camera/setup label<input name="camera" maxlength="120" placeholder="Optional; use the same label for repeats"></label>
<label>Capture date (optional)<input name="captured_at" type="datetime-local"></label>
<details><summary>Optionally keep selected photos and review points</summary><p>Leave all unchecked for report only. Saved annotations must be reviewed again after restoring.</p><div data-history-photos></div></details>
<button>Save to my history</button></fieldset></form><a href="/history">Open private history</a>
<details><summary>Local illustrative preview</summary><p>Adjustable original geometric assets. A styling experiment, never a measurement, fit prescription or prediction.</p>
<form method="post" action="/history/locked" data-preview-form><fieldset disabled data-history-controls><label>Source photo<select name="photo" data-preview-photos></select></label>
<label>Overlay<select name="style"><option value="eyewear">Eyewear</option><option value="hair">Hair silhouette</option><option value="reviewed">Original / reviewed frontal points</option></select></label>
<label>Horizontal position<input name="x" type="range" min="0" max="1" step="0.01" value="0.5"></label>
<label>Vertical position<input name="y" type="range" min="0" max="1" step="0.01" value="0.4"></label>
<label>Scale<input name="scale" type="range" min="0.1" max="0.8" step="0.01" value="0.4"></label>
<label><input name="original" type="checkbox"> Show original without overlay</label><button>Render local preview</button></fieldset></form>
<div data-preview-output></div><p>Previews have a separate internal type and SVG exports are rejected by photo intake. Re-encoded outside images cannot be authenticated.</p></details></section>'''
