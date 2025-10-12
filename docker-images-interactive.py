#!/usr/bin/env python3
import curses
import subprocess
import json
import time
from typing import TypedDict, List, Tuple


# Docker image fields from 'docker images --format {{json .}}'
class ImageInfo(TypedDict):
    ID: str
    Repository: str
    Tag: str
    Digest: str
    CreatedSince: str
    CreatedAt: str
    Size: str


# Docker container fields from 'docker ps -a --format {{json .}}'
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


# Utility to run 'docker images --format {{json .}}' and return list of ImageInfo
def list_docker_images() -> list[ImageInfo]:
    result = subprocess.run([
        'docker', 'images', '--format', '{{json .}}'
    ], capture_output=True, text=True)

    return [json.loads(line) for line in result.stdout.strip().split('\n') if line]


# Utility to run 'docker ps -a --format {{json .}}' and return list of ContainerInfo
def list_docker_containers() -> list[ContainerInfo]:
    result = subprocess.run([
        'docker', 'ps', '-a', '--format', '{{json .}}'
    ], capture_output=True, text=True)

    return [json.loads(line) for line in result.stdout.strip().split('\n') if line]


# Function that returns list of (ImageInfo, List[ContainerInfo]) tuples
def get_images_with_containers() -> List[Tuple[ImageInfo, List[ContainerInfo]]]:
    images = list_docker_images()
    containers = list_docker_containers()
    repo_tag_to_image_id = {f"{img['Repository']}:{img['Tag']}": img['ID'] for img in images}

    # Create a mapping from image ID to containers using that image
    image_id_to_containers: dict[str, list[ContainerInfo]] = {}
    for container in containers:
        image_ref = container['Image']  # Can be repo:tag or the image ID
        image_id = repo_tag_to_image_id.get(image_ref, image_ref)
        image_id_to_containers.setdefault(image_id, []).append(container)

    # Create result list with image info and associated containers
    return [(img, image_id_to_containers.get(img['ID'], [])) for img in images]


def left_ellipsis(value: str, sz: int):
    return ('…' + value[-(sz-1):]) if len(value) > sz else value


# Helper to find the best matching image index after refresh
def find_best_matching_image(
    images: List[Tuple[ImageInfo, List[ContainerInfo]]],
    old_id: str,
    old_repo: str,
    old_tag: str
) -> int:
    # 1. Same id, repo, tag
    for i, (img, _) in enumerate(images):
        if img['ID'] == old_id and img['Repository'] == old_repo and img['Tag'] == old_tag:
            return i
    # 2. Same id, repo
    for i, (img, _) in enumerate(images):
        if img['ID'] == old_id and img['Repository'] == old_repo:
            return i
    # 3. Same id
    for i, (img, _) in enumerate(images):
        if img['ID'] == old_id:
            return i
    return 0


def main(stdscr: curses.window):
    curses.curs_set(0)

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        # Initialize color pair 1 for normal text with default foreground and background colors
        curses.init_pair(1, -1, -1)  # -1 means default terminal colors

    k = 0
    selected = 0
    confirm_delete = False
    image_container_pairs = get_images_with_containers()

    # Scroll-related variables
    scroll_offset = 0
    max_display_lines = 0

    while True:
        stdscr.clear()
        max_y, _ = stdscr.getmaxyx()
        max_display_lines = max_y - 2  # Reserve 2 lines for header and instructions

        # Calculate scroll offset to keep selected item visible
        if selected < scroll_offset:
            scroll_offset = selected
        elif selected >= scroll_offset + max_display_lines:
            scroll_offset = selected - max_display_lines + 1

        # Ensure scroll offset is not negative
        scroll_offset = max(0, scroll_offset)

        stdscr.addstr(0, 0, "ID           REPOSITORY                       TAG              SIZE       CREATED          USED")

        # Display only the visible range of images
        for idx, (img, containers) in enumerate(image_container_pairs[scroll_offset:scroll_offset+max_display_lines]):
            display_idx = idx + scroll_offset
            marker = '*' if len(containers) > 0 else ' '
            line = f"{img['ID'][:12]:12} {left_ellipsis(img['Repository'], 32):32} {left_ellipsis(img['Tag'], 16):16} {img['Size']:10} {img['CreatedSince']:16} {marker}"
            if display_idx == selected:
                stdscr.addstr(1+idx, 0, line, curses.A_REVERSE)
            else:
                stdscr.addstr(1+idx, 0, line)

        if confirm_delete:
            image = image_container_pairs[selected][0]
            stdscr.addstr(max_y-1, 0, f"Delete image {image['Repository']}:{image['Tag']}? (y/n)")
        else:
            stdscr.addstr(max_y-1, 0, "(q: quit, d: delete, r/F5: refresh)")
            stdscr.refresh()

        k = stdscr.getch()

        if confirm_delete and k in [ord('y'), ord('Y')]:
            # Only delete if not used
            img, containers = image_container_pairs[selected]
            if len(containers) == 0:  # No containers using this image
                subprocess.run(['docker', 'rmi', img['ID']])
                image_container_pairs = get_images_with_containers()
                selected = min(selected, len(image_container_pairs)-1)
                # Adjust scroll offset after deletion
                if selected < scroll_offset:
                    scroll_offset = selected
                elif selected >= scroll_offset + max_display_lines:
                    scroll_offset = selected - max_display_lines + 1

            confirm_delete = False
        else:
            confirm_delete = False

            if k == curses.KEY_UP:
                selected = max(0, selected-1)
            elif k == curses.KEY_DOWN:
                selected = min(len(image_container_pairs)-1, selected+1)
            elif k == ord('d'):
                # Only allow delete if not used
                img, containers = image_container_pairs[selected]
                if len(containers) == 0:  # No containers using this image
                    confirm_delete = True
            elif k == ord('q'):
                break
            elif k == ord('r') or k == curses.KEY_F5:  # Refresh
                old_selected_image = image_container_pairs[selected][0]
                rel_pos = selected - scroll_offset

                image_container_pairs = get_images_with_containers()

                selected = find_best_matching_image(
                    image_container_pairs,
                    old_selected_image['ID'],
                    old_selected_image['Repository'],
                    old_selected_image['Tag']
                )

                # Recompute scroll_offset to keep the selected image at the same relative position
                scroll_offset = selected - rel_pos
                if scroll_offset < 0:
                    scroll_offset = 0
                elif scroll_offset > max(0, len(image_container_pairs) - max_display_lines):
                    scroll_offset = max(0, len(image_container_pairs) - max_display_lines)
        time.sleep(0.05)


if __name__ == "__main__":
    curses.wrapper(main)
