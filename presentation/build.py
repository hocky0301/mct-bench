"""Re-embed narration.json and the robot SVG into the standalone HTML.

Run: python3 presentation/build.py
Slide layout and visible headings are edited directly in presentation.html.
The robot illustration is a self-drawn geometric SVG (assets/robot.svg),
no external or generated image material.
Python standard library only.
"""
from pathlib import Path
import base64
import json
import re


def main():
    here = Path(__file__).resolve().parent
    scripts = json.loads((here / "narration.json").read_text(encoding="utf-8"))
    if not isinstance(scripts, list) or len(scripts) != 10:
        raise ValueError("narration.json must contain exactly ten slides")
    for slide in scripts:
        if not isinstance(slide.get("narration"), str) or not slide["narration"].strip():
            raise ValueError("Each slide needs narration text")
        if not isinstance(slide.get("sourceURLs"), list):
            raise ValueError("Each slide needs a sourceURLs list")
        if not all(isinstance(url, str) and url.startswith("https://") for url in slide["sourceURLs"]):
            raise ValueError("Source links must use HTTPS")
    html_path = here / "presentation.html"
    html = html_path.read_text(encoding="utf-8")
    encoded = json.dumps(scripts, ensure_ascii=False).replace("</", "<\\/")
    html, count = re.subn(
        r'(<script id="script-data" type="application/json">).*?(</script>)',
        lambda match: match[1] + encoded + match[2], html, flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Expected exactly one script-data block")
    image = base64.b64encode((here / "assets/robot.svg").read_bytes()).decode("ascii")
    html, count = re.subn(
        r'(<img class="robot" src=")data:image/[a-z+]+;base64,[^"]+(" alt=)',
        lambda match: match[1] + "data:image/svg+xml;base64," + image + match[2], html,
    )
    if count != 1:
        raise ValueError("Expected exactly one robot image")
    html_path.write_text(html, encoding="utf-8")
    transcript = ["# プレゼンテーションの読み上げ台本\n"]
    for number, slide in enumerate(scripts, 1):
        transcript.extend([f'## {number:02} / {slide["title"]}\n', slide["narration"] + "\n"])
    (here / "narration.md").write_text("\n".join(transcript), encoding="utf-8")
    print("Updated presentation.html and narration.md")


if __name__ == "__main__":
    main()
