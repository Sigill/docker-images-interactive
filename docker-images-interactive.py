#!/usr/bin/env python3
import curses
import subprocess
import json
import time
from typing import TypedDict


# Docker image fields from 'docker images --format {{json .}}'
class ImageInfo(TypedDict):
    ID: str
    Repository: str
    Tag: str
    Digest: str
    CreatedSince: str
    CreatedAt: str
    Size: str


# Utility to run 'docker images --format {{json .}}' and return list of ImageInfo
def list_docker_images() -> list[ImageInfo]:
    result = subprocess.run([
        'docker', 'images', '--format', '{{json .}}'
    ], capture_output=True, text=True)

    return [json.loads(line) for line in result.stdout.strip().split('\n') if line]


# Docker image fields from 'docker ps -a --format {{json .}}'
class ContainerInfo(TypedDict):
    ID: str
    Image: str
    Command: str
    CreatedAt: str
    RunningFor: str
    Ports: str
    Status: str
    Size: str
    Names: str
    Labels: str
    Mounts: str


# Utility to run 'docker ps -a --format {{json .}}' and return list of ContainerInfo
def list_docker_containers() -> list[ContainerInfo]:
    result = subprocess.run([
        'docker', 'ps', '-a', '--format', '{{json .}}'
    ], capture_output=True, text=True)

    return [json.loads(line) for line in result.stdout.strip().split('\n') if line]


# Helper to get images used by containers
def get_used_image_ids() -> set[str]:
    containers = list_docker_containers()
    images = list_docker_images()
    # Map image repo:tag and image ID for lookup
    repo_tag_to_image_id = {f"{img['Repository']}:{img['Tag']}": img['ID'] for img in images}
    image_id_set = {img['ID'] for img in images}
    used_image_ids: set[str] = set()
    for container in containers:
        image_ref = container['Image']
        # Try to match by repo:tag first, then by ID
        if image_ref in repo_tag_to_image_id:
            used_image_ids.add(repo_tag_to_image_id[image_ref])
        elif image_ref in image_id_set:
            used_image_ids.add(image_ref)
    return used_image_ids


def main(stdscr: curses.window):
    curses.curs_set(0)
    k = 0
    selected = 0
    confirm_delete = False
    images = list_docker_images()
    used_image_ids = get_used_image_ids()
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Docker Images Browser (q: quit, d: delete)")
        stdscr.addstr(1, 0, "ID         REPOSITORY      TAG        SIZE      CREATED         USED")
        for idx, img in enumerate(images):
            marker = '*' if img['ID'] in used_image_ids else ' '
            line = f"{img['ID'][:12]:12} {img['Repository'][:14]:14} {img['Tag'][:10]:10} {img['Size']:10} {img['CreatedSince']:14} {marker}"
            if idx == selected:
                stdscr.addstr(2+idx, 0, line, curses.A_REVERSE)
            else:
                stdscr.addstr(2+idx, 0, line)
        if confirm_delete:
            stdscr.addstr(2+len(images)+1, 0, f"Delete image {images[selected]['Repository']}:{images[selected]['Tag']}? (y/n)")
        stdscr.refresh()
        k = stdscr.getch()
        if confirm_delete:
            if k in [ord('y'), ord('Y')]:
                # Only delete if not used
                if images[selected]['ID'] not in used_image_ids:
                    subprocess.run(['docker', 'rmi', images[selected]['ID']])
                    images = list_docker_images()
                    used_image_ids = get_used_image_ids()
                    selected = min(selected, len(images)-1)
                confirm_delete = False
            elif k in [ord('n'), ord('N')]:
                confirm_delete = False
        else:
            if k == curses.KEY_UP:
                selected = max(0, selected-1)
            elif k == curses.KEY_DOWN:
                selected = min(len(images)-1, selected+1)
            elif k == ord('d'):
                # Only allow delete if not used
                if images[selected]['ID'] not in used_image_ids:
                    confirm_delete = True
            elif k == ord('q'):
                break
        time.sleep(0.05)


if __name__ == "__main__":
    curses.wrapper(main)
