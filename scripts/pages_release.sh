#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    record)
        cd "$2"
        [[ $3 =~ ^[0-9a-f]{40}$ ]]
        printf '%s\n' "$3" > source-commit.txt
        sha256sum source-commit.txt index.html out/viewer.js out/viewer_bg.wasm \
            project/index.html project/publication.json project/static/papers/publication-run.json \
            project/static/papers/bevy_zeroverse.pdf project/fisheye/whitepaper.pdf \
            project/fisheye/fisheye_cameras.pdf > deployment.sha256
        sha256sum deployment.sha256 | cut -d' ' -f1
        ;;
    verify)
        site=${2%/}
        expected=$3
        [[ $expected =~ ^[0-9a-f]{64}$ ]]
        receipt=$(mktemp)
        trap 'rm -f "$receipt"' EXIT
        fetch=(curl --fail --silent --show-error --location --connect-timeout 15 --max-time 180)
        # The Pages CDN can still serve the previous deployment immediately after promotion.
        for attempt in {1..18}; do
            if "${fetch[@]}" "$site/deployment.sha256?release=$expected&attempt=$attempt" -o "$receipt" &&
                [[ $(sha256sum "$receipt" | cut -d' ' -f1) == "$expected" ]]; then
                break
            fi
            if (( attempt == 18 )); then
                echo 'The live Pages receipt did not match the artifact approved for deployment.' >&2
                exit 1
            fi
            sleep 10
        done
        while read -r digest path; do
            for attempt in 1 2 3; do
                if actual=$("${fetch[@]}" "$site/$path?release=$expected&attempt=$attempt" | sha256sum) &&
                    [[ ${actual%% *} == "$digest" ]]; then
                    break
                fi
                if (( attempt == 3 )); then
                    printf 'Live file failed its artifact checksum: %s\n' "$path" >&2
                    exit 1
                fi
                sleep 2
            done
        done < "$receipt"
        echo 'Verified the live source revision, viewer, gallery, receipts and papers.'
        ;;
    *)
        echo 'Usage: pages_release.sh record WWW_DIR COMMIT | verify SITE_URL EXPECTED_RECEIPT_SHA256' >&2
        exit 2
        ;;
esac
