#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "torrentfile",
# ]
# ///

import fnmatch
import hashlib
import os
import time
import urllib.parse

import click
import pyben
from torrentfile.torrent import TorrentFileHybrid

DEFAULT_PIECE_SIZE = 2 * 1024 * 1024  # 2 MiB

# Tracker list from trackers_best.txt
# https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best.txt
TRACKERS = [
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://open.demonii.com:1337/announce",
    "udp://open.stealth.si:80/announce",
    "udp://tracker.torrent.eu.org:451/announce",
    "udp://wepzone.net:6969/announce",
    "udp://utracker.ghostchu-services.top:6969/announce",
    "udp://udp.tracker.projectk.org:23333/announce",
    "udp://tracker.tvunderground.org.ru:3218/announce",
    "udp://tracker.tryhackx.org:6969/announce",
    "udp://tracker.theoks.net:6969/announce",
    "udp://tracker.t-1.org:6969/announce",
    "udp://tracker.srv00.com:6969/announce",
    "udp://tracker.qu.ax:6969/announce",
    "udp://tracker.playground.ru:6969/announce",
    "udp://tracker.opentorrent.top:6969/announce",
    "udp://tracker.ixuexi.click:6969/announce",
    "udp://tracker.gmi.gd:6969/announce",
    "udp://tracker.fnix.net:6969/announce",
    "udp://tracker.filemail.com:6969/announce",
    "udp://tracker.ducks.party:1984/announce",
]


def make_magnet(torrent_path, tracker_count):
    meta = pyben.load(torrent_path)
    info = meta["info"]
    info_bencoded = pyben.dumps(info)

    v1_hash = hashlib.sha1(info_bencoded).hexdigest()
    v2_hash = hashlib.sha256(info_bencoded).hexdigest()
    name = urllib.parse.quote(info["name"])

    trackers = TRACKERS[:tracker_count]
    tr_params = "&".join(f"tr={urllib.parse.quote(t)}" for t in trackers)

    return f"magnet:?xt=urn:btih:{v1_hash}&xt=urn:btmh:1220{v2_hash}&dn={name}&{tr_params}"


@click.command()
@click.option("-i", "--input-directory", required=True, type=click.Path(exists=True),
              help="The folder containing the downloaded models.")
@click.option("-o", "--output-directory", required=True, type=click.Path(),
              help="The folder where generated torrent files will be saved.")
@click.option("--filter", "filter_pattern", default="*", show_default=True,
              help="Glob pattern for folder names (e.g. '*AWQ*').")
@click.option("--piece-size", default=DEFAULT_PIECE_SIZE, show_default=True, type=int,
              help="Piece size in bytes.")
@click.option("--print-magnets", is_flag=True, help="Print magnet links to stdout.")
@click.option("--magnets-file", default=None, type=click.Path(),
              help="Write magnet links to this file.")
@click.option("--magnet-tracker-count", default="5", show_default=True,
              help="Number of trackers to include in magnet links, or 'all'.")
def main(input_directory, output_directory, filter_pattern, piece_size,
         print_magnets, magnets_file, magnet_tracker_count):
    """Generate hybrid v1+v2 torrent files for AI models."""
    os.makedirs(output_directory, exist_ok=True)

    if magnet_tracker_count.lower() == "all":
        tracker_count = len(TRACKERS)
    else:
        tracker_count = min(int(magnet_tracker_count), len(TRACKERS))

    folders = sorted(
        f for f in os.listdir(input_directory)
        if os.path.isdir(os.path.join(input_directory, f))
        and fnmatch.fnmatch(f, filter_pattern)
    )

    if not folders:
        click.echo("No matching folders found.")
        return

    click.echo(f"Found {len(folders)} folder(s) to process with {len(TRACKERS)} trackers.")
    start_time = time.time()
    magnet_lines = []

    for i, folder in enumerate(folders, start=1):
        content_path = os.path.join(input_directory, folder)
        output_file = os.path.join(output_directory, f"{folder}.torrent")

        click.echo(f"[{i}/{len(folders)}] {folder}")
        folder_start = time.time()

        t = TorrentFileHybrid(
            path=content_path,
            announce=TRACKERS,
            piece_length=piece_size,
            progress=0,
        )
        t.write(output_file)

        if print_magnets or magnets_file:
            magnet = make_magnet(output_file, tracker_count)
            if print_magnets:
                click.echo(f"  {magnet}")
            if magnets_file:
                magnet_lines.append(f"# {folder}\n{magnet}\n")

        elapsed = time.time() - folder_start
        click.echo(f"  Done in {elapsed:.1f}s -> {output_file}")

    if magnets_file and magnet_lines:
        with open(magnets_file, "a") as f:
            f.write("\n".join(magnet_lines))
        click.echo(f"Magnet links written to {magnets_file}")

    total = time.time() - start_time
    click.echo(f"Total time: {total:.1f}s")


if __name__ == "__main__":
    main()
