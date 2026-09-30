"""Browser assertions for matched baseline views; called by validate_project_page."""
from urllib.parse import urljoin


def wait_selection(page, seed, baseline, mode):
    value = f"{seed}:{baseline:.2f}:{mode}"
    page.wait_for_function("s => document.getElementById('baseline-explorer').dataset.selection === s", arg=value)
    assert page.locator("#baseline-explorer").get_attribute("aria-busy") == "false"


def check_views(page, room, level, mode):
    wait_selection(page, room["seed"], level["baseline"], mode)
    assert page.locator("#baseline-plan").get_attribute("src").endswith(level["plan"])
    assert page.locator("#baseline-distance").inner_text() == f'{level["mean_reference_m"]:.2f} m'
    assert page.locator("#baseline-shared").inner_text() == f'{level["shared_any"]*100:.1f}%'
    for c, view in enumerate(level["views"]):
        tile = page.locator("#baseline-views figure").nth(c)
        assert tile.locator("img").get_attribute("src").endswith(view[mode])
        assert tile.locator("a").get_attribute("href").endswith(view[mode])
        assert tile.locator("img").evaluate("i => i.complete && i.naturalWidth === 768 && i.naturalHeight === 480")
        assert tile.locator(".baseline-fov").inner_text() == f'{view["camera"]["fovy_degrees"]:.1f}° FOV'
    assert page.locator("#baseline-contact-image").get_attribute("src").endswith(room["contact"])


def validate(page, url, output):
    page.wait_for_selector('#baseline-explorer[data-ready="true"]')
    data = page.request.get(urljoin(url,"static/media/baseline/gallery.json")).json()
    cases = 0
    for r, room in enumerate(data["rooms"]):
        page.select_option("#baseline-room",str(r))
        for level in room["levels"]:
            page.locator("#baseline-slider").fill(f'{level["baseline"]:g}')
            for mode in ("rgb","shared"):
                page.click(f'[data-baseline-mode="{mode}"]')
                check_views(page,room,level,mode)
                cases += 1
        page.locator(".baseline-contact summary").click()
        page.wait_for_function("document.getElementById('baseline-contact-image').naturalWidth === 1808")
        page.locator(".baseline-contact summary").click()
    # Keyboard: Home/End and arrows span actual captured settings.
    page.locator("#baseline-slider").focus()
    page.keyboard.press("Home")
    wait_selection(page,24000,0,"shared")
    page.keyboard.press("ArrowRight")
    wait_selection(page,24000,.25,"shared")
    page.keyboard.press("End")
    wait_selection(page,24000,1,"shared")
    assert page.locator("#baseline-slider").get_attribute("aria-valuetext").startswith("1.00")
    # Several changes in one JS task must commit only the final complete set.
    page.evaluate("""() => {
      const slider = document.getElementById('baseline-slider');
      slider.value = 0; slider.dispatchEvent(new Event('input'));
      const room = document.getElementById('baseline-room');
      room.value = '0'; room.dispatchEvent(new Event('change'));
      document.querySelector('[data-baseline-mode="rgb"]').click();
      slider.value = .75; slider.dispatchEvent(new Event('input'));
    }""")
    check_views(page,data["rooms"][0],data["rooms"][0]["levels"][3],"rgb")
    page.click('[data-baseline="0.5"]')
    wait_selection(page,24005,.5,"rgb")
    page.locator("#baseline").screenshot(path=str(output/"baseline-desktop.png"))
    return dict(configurations=10,view_display_cases=cases*4,keyboard_slider=True,
                rapid_switch_identity=True,all_images_decoded=True)


def responsive(page, width, output):
    page.wait_for_selector('#baseline-explorer[data-ready="true"]')
    page.click('[data-baseline="1"]')
    wait_selection(page,24005,1,"rgb")
    assert page.locator(".baseline-views").evaluate(
        "e => getComputedStyle(e).gridTemplateColumns.split(' ').length") == (1 if width < 640 else 2)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.locator("#baseline").screenshot(path=str(output/f"baseline-{width}.png"))


def without_javascript(browser, url):
    context = browser.new_context(java_script_enabled=False,viewport={"width":390,"height":844})
    page = context.new_page()
    page.goto(url,wait_until="networkidle")
    assert page.locator("#baseline-slider").is_disabled()
    assert page.locator("#baseline-views img").count() == 4
    page.locator(".baseline-contact summary").click()
    assert page.locator("#baseline-contact-image").is_visible()
    context.close()
