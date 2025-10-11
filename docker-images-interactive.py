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

    # Create a mapping from image ID to containers using that image
    image_id_to_containers: dict[str, list[ContainerInfo]] = {}
    for container in containers:
        image_id = container['Image']
        if image_id not in image_id_to_containers:
            image_id_to_containers[image_id] = []
        image_id_to_containers[image_id].append(container)

    # Map repository:tag to image ID for lookup
    repo_tag_to_image_id = {f"{img['Repository']}:{img['Tag']}": img['ID'] for img in images}

    # Create result list with image info and associated containers
    result: list[Tuple[ImageInfo, List[ContainerInfo]]] = []
    for img in images:
        # Get the image ID (either from the image itself or from repo:tag mapping)
        image_id = img['ID']
        if image_id in repo_tag_to_image_id:
            # If we have a repo:tag mapping, use it to get the actual image ID
            image_id = repo_tag_to_image_id[image_id]

        # Get containers using this image
        containers_for_image = image_id_to_containers.get(image_id, [])
        result.append((img, containers_for_image))

    return result


def main(stdscr: curses.window):
    curses.curs_set(0)
    k = 0
    selected = 0
    confirm_delete = False
    # Use the new function instead of separate calls
    image_container_pairs = get_images_with_containers()
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Docker Images Browser (q: quit, d: delete)")
        stdscr.addstr(1, 0, "ID         REPOSITORY      TAG        SIZE      CREATED         USED")
        for idx, (img, containers) in enumerate(image_container_pairs):
            marker = '*' if len(containers) > 0 else ' '
            line = f"{img['ID'][:12]:12} {img['Repository'][:14]:14} {img['Tag'][:10]:10} {img['Size']:10} {img['CreatedSince']:14} {marker}"
            if idx == selected:
                stdscr.addstr(2+idx, 0, line, curses.A_REVERSE)
            else:
                stdscr.addstr(2+idx, 0, line)
        if confirm_delete:
            stdscr.addstr(2+len(image_container_pairs)+1, 0, f"Delete image {image_container_pairs[selected][0]['Repository']}:{image_container_pairs[selected][0]['Tag']}? (y/n)")
        stdscr.refresh()
        k = stdscr.getch()
        if confirm_delete:
            if k in [ord('y'), ord('Y')]:
                # Only delete if not used
                img, containers = image_container_pairs[selected]
                if len(containers) == 0:  # No containers using this image
                    subprocess.run(['docker', 'rmi', img['ID']])
                    image_container_pairs = get_images_with_containers()
                    selected = min(selected, len(image_container_pairs)-1)
                confirm_delete = False
            elif k in [ord('n'), ord('N')]:
                confirm_delete = False
        else:
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
        time.sleep(0.05)


if __name__ == "__main__":
    curses.wrapper(main)
