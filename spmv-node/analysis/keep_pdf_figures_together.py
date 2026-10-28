"""Keep each numbered caption with its figure in the renderer's temporary HTML."""

from pathlib import Path
import re
import sys

html_path = Path(sys.argv[1])
html = html_path.read_text()
pattern = re.compile(
    r'(<p><img [^>]+></p>)\n(<p><strong>Figure \d+\.</strong>.*?</p>)',
    re.DOTALL,
)
html, count = pattern.subn(r'<div class="figure-block">\1\n\2</div>', html)
if count != 10:
    raise SystemExit(f"Expected 10 figure/caption pairs; found {count}")
html = html.replace(
    "</style>",
    ".figure-block { break-inside: avoid; page-break-inside: avoid; margin: 8px 0 12px; }"
    ".figure-block p { margin: 3px 0; }"
    "</style>",
    1,
)
html_path.write_text(html)
