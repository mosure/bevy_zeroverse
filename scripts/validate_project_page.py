#!/usr/bin/env python3
"""Browser check of actual gallery identities, controls, media, links and layout."""
import argparse
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8770/project/")
    parser.add_argument("--output", type=Path, default=Path("out/project_page/browser"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures, cases = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                                    args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
        page = context.new_page()
        page.on("pageerror", lambda error: failures.append(str(error)))
        page.on("response", lambda response: failures.append(f"HTTP {response.status}: {response.url}")
                if response.status >= 400 else None)
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_selector('#comparison[data-ready="true"]')
        data = page.request.get(urljoin(args.url, "static/media/gallery.json")).json()
        assert page.locator("#room-video").evaluate("v => v.paused"), "reduced motion must suppress autoplay"
        page.screenshot(path=str(args.output / "desktop.png"))
        for s, scene in enumerate(data["scenes"]):
            page.select_option("#scene-select", str(s))
            for c in range(4):
                page.click(f'[data-camera="{c}"]')
                for mode, image in scene["frames"][0]["views"][c]["images"].items():
                    if mode == "peers":
                        continue
                    page.click(f'[data-mode="{mode}"]')
                    page.wait_for_function("name => document.getElementById('annotation-image').src.endsWith(name)", arg=image)
                    assert page.locator("#rgb-image").get_attribute("src").endswith(scene["frames"][0]["views"][c]["images"]["color"])
                    assert page.locator("#annotation-image").evaluate("i => i.complete && i.naturalWidth === 768")
                    cases.append(dict(scene=scene["seed"], camera=c, mode=mode))
        page.click('[data-mode="co_visibility"]')
        page.locator("#peer-buttons button", has_text="Camera 0").click()
        page.wait_for_function("document.getElementById('annotation-image').src.endsWith('peer0.webp')")
        assert page.locator("#peer-buttons button", has_text="Camera 3").is_disabled()
        page.select_option("#time-select", "2")
        page.click('[data-mode="optical_flow"]')
        page.wait_for_function("document.getElementById('mode-description').textContent.startsWith('Terminal')")
        page.select_option("#time-select", "1")
        page.wait_for_function("document.getElementById('annotation-image').src.includes('-t1-')")
        assert "t=0.10 to t=0.20" in page.locator("#mode-description").inner_text()
        page.locator("#reveal").fill("75")
        assert page.locator("#comparison").evaluate("e => e.style.getPropertyValue('--split')") == "75%"
        page.locator("#reveal").focus()
        page.keyboard.press("ArrowLeft")
        assert page.locator("#reveal").input_value() == "74"
        # Final matched-view screenshot, with every response checked for failures.
        page.select_option("#scene-select", "0"); page.select_option("#time-select", "0")
        page.click('[data-camera="0"]'); page.click('[data-mode="co_visibility"]')
        page.locator("#peer-buttons button", has_text="All cameras").click()
        page.locator("#reveal").fill("50")
        page.wait_for_function("document.getElementById('annotation-image').src.endsWith('s24005-t0-c0-co_visibility.png')")
        page.locator(".explorer").screenshot(path=str(args.output / "matched-annotations.png"))
        # Every local asset link must resolve, including PDF, source ZIP and JSON.
        links = page.locator("a[href],img[src],source[src],link[href],script[src]").evaluate_all(
            "nodes => [...new Set(nodes.map(n => n.getAttribute('href') || n.getAttribute('src')))]")
        checked = []
        for link in links:
            if link.startswith("#"):
                assert page.locator(link).count(), f"missing anchor {link}"
                continue
            target = urljoin(args.url, link)
            if urlparse(target).netloc != urlparse(args.url).netloc:
                continue
            response = page.request.get(target)
            assert response.ok, (target, response.status)
            checked.append(target)
            if target.endswith(".pdf"):
                assert response.body().startswith(b"%PDF-")
        for video in ["room-video", "motion-video"]:
            page.locator(f"#{video}").evaluate("v => {v.load(); v.muted = true; return v.play();}")
            page.wait_for_function("id => document.getElementById(id).currentTime > 0.15", arg=video)
            assert page.locator(f"#{video}").evaluate("v => v.videoWidth === 1280 && v.duration === 6")
            page.locator(f"#{video}").evaluate("v => v.pause()")
        for width in [390, 768]:
            mobile = context.new_page()
            mobile.set_viewport_size({"width": width, "height": 844})
            mobile.on("pageerror", lambda error: failures.append(str(error)))
            mobile.goto(args.url, wait_until="networkidle")
            mobile.wait_for_selector('#comparison[data-ready="true"]')
            assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), width
            expected_columns = 1 if width < 640 else 2
            assert mobile.locator(".plot-grid").first.evaluate(
                "e => getComputedStyle(e).gridTemplateColumns.split(' ').length") == expected_columns
            mobile.select_option("#scene-select", "1")
            mobile.click('[data-camera="2"]'); mobile.click('[data-mode="depth"]')
            mobile.wait_for_function("document.getElementById('annotation-image').src.endsWith('s24000-t0-c2-depth.png')")
            mobile.evaluate("window.scrollTo(0, 0)")
            mobile.screenshot(path=str(args.output / f"mobile-{width}.png"))
            if width == 390:
                mobile.locator(".plot-grid").first.locator("img").evaluate_all("images => images.forEach(i => i.loading = 'eager')")
                mobile.wait_for_function("[...document.querySelector('.plot-grid').querySelectorAll('img')].every(i => i.complete && i.naturalWidth > 0)")
                mobile.set_viewport_size({"width": width, "height": 2400})
                mobile.locator(".plot-grid").first.screenshot(path=str(args.output / "mobile-charts.png"))
            mobile.close()
        autoplay = context.new_page()
        autoplay.emulate_media(reduced_motion="no-preference")
        autoplay.goto(args.url, wait_until="networkidle")
        autoplay.wait_for_function("document.getElementById('room-video').currentTime > 0.15")
        autoplay.close()
        assert not failures, failures
        report = dict(passed=True, matched_mode_cases=len(cases), cases=cases, checked_local_links=len(checked),
                      media=["4-camera traversal: 120 frames at 20 Hz", "2-camera ARDY motion: 120 frames at 20 Hz"],
                      responsive_widths=[390, 768, 1440], reduced_motion_respected=True,
                      visible_teaser_autoplay=True,
                      keyboard_reveal=True, camera_membership_selection=True, terminal_flow_checked=True,
                      errors=failures)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
