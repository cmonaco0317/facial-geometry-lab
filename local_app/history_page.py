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
<section><h2>Saved reports</h2><p>100 reports maximum · 6 MB per report · 200 MB total. Numerical-only saves omit photos and point geometry. Opted-in full reports may retain crops, source points and earlier saved contexts without original photos.</p>
<button type="button" data-history-refresh>Refresh history</button><div data-history-list></div>
<label>Before report<select data-compare-before><option value="">Choose the earlier report</option></select></label>
<label>After report<select data-compare-after><option value="">Choose the later report</option></select></label>
<button type="button" data-history-compare>Compare before and after</button></section>
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
<h2>Keep this report?</h2><p>This session is temporary. Saving creates a separate encrypted numerical snapshot. You can also retain the full report with all its contained source geometry, crops and earlier saved contexts, and separately select original photos below.</p>
<p data-history-status role="status">Open the app to enable private history.</p>
<form method="post" action="/history/locked" data-history-save><fieldset disabled data-history-controls><label>Local profile name<input name="profile" maxlength="120" required placeholder="Choose a name; no face matching"></label>
<label>Report name<input name="name" maxlength="120" placeholder="Optional label"></label>
<label>Capture conditions<select name="protocol"><option value="unspecified">Not recorded</option><option value="casual">Casual capture</option><option value="controlled">Controlled repeat setup</option><option value="mixed">Mixed conditions</option></select></label>
<label>Camera/setup label<input name="camera" maxlength="120" placeholder="Optional; use the same label for repeats"></label>
<label>Capture date (optional)<input name="captured_at" type="datetime-local"></label>
<p>Record conditions for this set. These labels are supplied by you, not measured pose or a verified repeat protocol.</p>
<label>Pose<select name="capture_pose"><option value="unknown">Not recorded</option><option value="straight_on">Straight on</option><option value="true_profile">True profile</option><option value="turned">Turned or tilted</option><option value="mixed">Mixed</option></select></label>
<label>Expression<select name="capture_expression"><option value="unknown">Not recorded</option><option value="neutral">Neutral</option><option value="smile">Smile</option><option value="other">Other</option><option value="mixed">Mixed</option></select></label>
<label>Lighting<select name="capture_lighting"><option value="unknown">Not recorded</option><option value="even">Even</option><option value="uneven">Uneven</option><option value="mixed">Mixed</option></select></label>
<label>Camera distance/setup<input name="capture_distance" maxlength="120" placeholder="Optional; include distance and framing"></label>
<label><input name="keep_context" type="checkbox">Keep the full report, source point geometry and crops, earlier correction/reviewer notes and revision history, saved question contexts and labeled illustrations.</label>
<p>The full report can contain earlier crops and illustrations even when no original photo is selected below. Leave unchecked for numerical history only. No empirical repeat-capture limits are available; arithmetic changes never establish physical progress.</p>
<details><summary>Optionally keep selected original photos and review points</summary><p>Leave all unchecked to omit original photos. Full-report retention above still includes its crops, points and saved contexts. Saved annotations must be reviewed again when restoring a new editable analysis.</p><div data-history-photos></div></details>
<button>Save to my history</button></fieldset></form><a href="/history">Open private history</a>
<details><summary>Local illustrative preview</summary><p>Adjustable original geometric assets. A styling experiment, never a measurement, fit prescription or prediction.</p>
<form method="post" action="/history/locked" data-preview-form><fieldset disabled data-history-controls><label>Source photo<select name="photo" data-preview-photos></select></label>
<label>Overlay<select name="style"><option value="eyewear">Eyewear</option><option value="hair">Hair silhouette</option><option value="reviewed">Original / reviewed frontal points</option></select></label>
<label>Horizontal position<input name="x" type="range" min="0" max="1" step="0.01" value="0.5"></label>
<label>Vertical position<input name="y" type="range" min="0" max="1" step="0.01" value="0.4"></label>
<label>Scale<input name="scale" type="range" min="0.1" max="0.8" step="0.01" value="0.4"></label>
<label><input name="original" type="checkbox"> Show original without overlay</label><button>Render local preview</button></fieldset></form>
<div data-preview-output></div><p>Previews have a separate internal type and SVG exports are rejected by photo intake. Re-encoded outside images cannot be authenticated.</p></details></section>'''
