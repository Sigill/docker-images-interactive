#!/usr/bin/env python3
import curses
import curses.ascii
import json
import subprocess
from typing import Any, List, Tuple, TypedDict


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


# Utility function to run a Docker command and parse the JSON output
def run_docker_command(command: List[str]) -> List[Any]:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in result.stdout.strip().split('\n') if line]


# Utility to run 'docker images --format {{json .}}' and return list of ImageInfo
def list_docker_images() -> List[ImageInfo]:
    return run_docker_command(['docker', 'images', '--format', '{{json .}}'])


# Utility to run 'docker ps -a --format {{json .}}' and return list of ContainerInfo
def list_docker_containers() -> List[ContainerInfo]:
    return run_docker_command(['docker', 'ps', '-a', '--format', '{{json .}}'])


# Function that returns list of (ImageInfo, List[ContainerInfo]) tuples
def get_images_with_containers() -> List[Tuple[ImageInfo, List[ContainerInfo]]]:
    images = list_docker_images()
    containers = list_docker_containers()
    repo_tag_to_image_id = {f"{img['Repository']}:{img['Tag']}": img['ID'] for img in images}

    # Create a mapping from image ID to containers using that image
    image_id_to_containers: dict[str, List[ContainerInfo]] = {}
    for container in containers:
        image_ref = container['Image']  # Can be repo:tag or the image ID
        image_id = repo_tag_to_image_id.get(image_ref, image_ref)
        image_id_to_containers.setdefault(image_id, []).append(container)

    # Create result list with image info and associated containers
    return [(img, image_id_to_containers.get(img['ID'], [])) for img in images]


def left_ellipsis(value: str, sz: int):
    return ('…' + value[-(sz-1):]) if len(value) > sz else value


def delete_image(img: ImageInfo):
    if img['Repository'] != '<none>' and img['Tag'] != '<none>':
        name = f"{img['Repository']}:{img['Tag']}"
    else:
        name = img['ID']

    # check=False, because docker rmi might fail if e.g. the image has already been removed from outside this script.
    subprocess.run(['docker', 'rmi', name], check=False)


def filter_images(image_pairs: List[Tuple[ImageInfo, List[ContainerInfo]]], keyword: str):
    """Filter image pairs based on keyword in repository:tag"""
    if not keyword:
        return image_pairs

    filtered: List[Tuple[ImageInfo, List[ContainerInfo]]] = []
    for img, containers in image_pairs:
        # Create the full repository:tag string for matching
        repo_tag = f"{img['Repository']}:{img['Tag']}"
        if keyword.lower() in repo_tag.lower():
            filtered.append((img, containers))

    return filtered


def display_editable_text(stdscr: curses.window, text: str, cursor_position: int, y: int, x: int):
    # Display the search keyword with cursor highlighting
    for i, char in enumerate(text):
        if i == cursor_position:
            # Highlight the character at cursor position
            stdscr.addch(y, x + i, char, curses.A_REVERSE)
        else:
            stdscr.addch(y, x + i, char)

    # If cursor is at the end, we need to add a reverse space to indicate cursor position
    if cursor_position == len(text):
        stdscr.addch(y, x + cursor_position, ' ', curses.A_REVERSE)


class ImageView:
    def __init__(self, stdscr: curses.window):
        self.stdscr = stdscr
        self.selected = 0
        self.confirm_delete_mode = False
        self.image_container_pairs = get_images_with_containers()
        self.scroll_offset = 0
        self.max_display_lines = 0
        self.search_mode = False
        self.search_keyword = ""
        self.saved_search_keyword = ""
        self.search_cursor_pos = 0

    def display(self) -> None:
        max_y, _ = self.stdscr.getmaxyx()
        self.max_display_lines = max_y - 2  # Reserve 2 lines for header and instructions

        # Calculate scroll offset to keep selected item visible
        if self.selected < self.scroll_offset:
            self.scroll_offset = self.selected
        elif self.selected >= self.scroll_offset + self.max_display_lines:
            self.scroll_offset = self.selected - self.max_display_lines + 1

        # Ensure scroll offset is not negative
        self.scroll_offset = max(0, self.scroll_offset)

        display_pairs = self.image_container_pairs
        # Apply filtering if in search mode
        used_search_keyword = self.search_keyword if self.search_mode else self.saved_search_keyword
        if used_search_keyword:
            display_pairs = filter_images(self.image_container_pairs, used_search_keyword)

        self.stdscr.addstr(0, 0, "ID           REPOSITORY                       TAG              SIZE       CREATED          USED")  # noqa: E501 # pylint: disable=line-too-long

        # Display only the visible range of images
        for idx, (img, containers) in enumerate(display_pairs[self.scroll_offset:self.scroll_offset+self.max_display_lines]):
            display_idx = idx + self.scroll_offset
            marker = '*' if len(containers) > 0 else ' '
            line = f"{img['ID'][:12]:12} {left_ellipsis(img['Repository'], 32):32} {left_ellipsis(img['Tag'], 16):16} {img['Size']:10} {img['CreatedSince']:16} {marker}"  # noqa: E501 # pylint: disable=line-too-long
            if display_idx == self.selected:
                self.stdscr.addstr(1+idx, 0, line, curses.A_REVERSE)
            else:
                self.stdscr.addstr(1+idx, 0, line)

        if self.search_mode:
            self.stdscr.addstr(max_y-1, 0, "Search: ")
            search_start_x = 8  # Position after "Search: "
            display_editable_text(self.stdscr, self.search_keyword, self.search_cursor_pos, max_y-1, search_start_x)
        elif self.confirm_delete_mode:
            image = display_pairs[self.selected][0]
            self.stdscr.addstr(max_y-1, 0, f"Delete image {image['Repository']}:{image['Tag']}? (y/n)")
        else:
            self.stdscr.addstr(max_y-1, 0, "(q: quit, d: delete, r/F5: refresh, g: first, G: last, /: search)")

        self.stdscr.refresh()

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def handle_input(self, k: int) -> bool:
        if self.search_mode:
            self.selected = 0

            if k in [curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
                # Validate and save the search keyword
                self.saved_search_keyword = self.search_keyword
                self.search_mode = False
                self.search_cursor_pos = 0
            elif k == curses.ascii.ESC:  # ESC key
                # Cancel search mode
                self.search_mode = False
                self.search_keyword = ""
                self.search_cursor_pos = 0
            elif k in [curses.KEY_BACKSPACE, curses.ascii.DEL, curses.ascii.BS]:
                # Backspace
                if self.search_cursor_pos > 0:
                    self.search_keyword = self.search_keyword[:self.search_cursor_pos-1] + self.search_keyword[self.search_cursor_pos:]
                    self.search_cursor_pos -= 1
            elif k == curses.KEY_LEFT:
                # Left arrow - move cursor left
                if self.search_cursor_pos > 0:
                    self.search_cursor_pos -= 1
            elif k == curses.KEY_RIGHT:
                # Right arrow - move cursor right
                if self.search_cursor_pos < len(self.search_keyword):
                    self.search_cursor_pos += 1
            elif k == curses.KEY_DC:  # Delete key
                # Delete character at cursor position
                if self.search_cursor_pos < len(self.search_keyword):
                    self.search_keyword = self.search_keyword[:self.search_cursor_pos] + self.search_keyword[self.search_cursor_pos+1:]
            elif 32 <= k <= 126:  # Printable characters
                # Add character to search keyword at cursor position
                self.search_keyword = self.search_keyword[:self.search_cursor_pos] + chr(k) + self.search_keyword[self.search_cursor_pos:]
                self.search_cursor_pos += 1
        elif self.confirm_delete_mode:
            if k in [ord('y'), ord('Y')]:
                # Only delete if not used
                img, containers = self.image_container_pairs[self.selected]
                if len(containers) == 0:  # No containers using this image
                    delete_image(img)
                    self.image_container_pairs = get_images_with_containers()
                    self.selected = min(self.selected, len(self.image_container_pairs)-1)
                    # Adjust scroll offset after deletion
                    if self.selected < self.scroll_offset:
                        self.scroll_offset = self.selected
                    elif self.selected >= self.scroll_offset + self.max_display_lines:
                        self.scroll_offset = self.selected - self.max_display_lines + 1

                self.confirm_delete_mode = False
            elif k in [ord('n'), ord('N')] or k == curses.ascii.ESC:
                self.confirm_delete_mode = False
        else:
            if k == curses.KEY_UP:
                self.selected = max(0, self.selected-1)
            elif k == curses.KEY_DOWN:
                self.selected = min(len(self.image_container_pairs)-1, self.selected+1)
            elif k == curses.KEY_NPAGE:  # Page Down
                if len(self.image_container_pairs) > 0:
                    # If not at the last visible item, jump to last visible
                    last_visible = min(self.scroll_offset + self.max_display_lines - 1, len(self.image_container_pairs) - 1)
                    if self.selected != last_visible:
                        self.selected = last_visible
                    else:
                        self.selected = min(self.selected + self.max_display_lines, len(self.image_container_pairs) - 1)
            elif k == curses.KEY_PPAGE:  # Page Up
                if len(self.image_container_pairs) > 0:
                    # If not at the first visible item, jump to first visible
                    first_visible = self.scroll_offset
                    if self.selected != first_visible:
                        self.selected = first_visible
                    else:
                        self.selected = max(self.selected - self.max_display_lines, 0)
            elif k == ord('g'):  # Select first image
                self.selected = 0
            elif k == ord('G'):  # Select last image
                self.selected = len(self.image_container_pairs) - 1
            elif k == ord('d'):
                # Only allow delete if not used
                img, containers = self.image_container_pairs[self.selected]
                if len(containers) == 0:  # No containers using this image
                    self.confirm_delete_mode = True
            elif k == ord('q'):
                return False
            elif k == ord('r') or k == curses.KEY_F5:  # Refresh
                self.image_container_pairs = get_images_with_containers()
                self.selected = 0
            if k == ord('/'):
                # Enter search mode
                self.search_mode = True
                self.search_keyword = self.saved_search_keyword  # Use the saved keyword when entering search mode
                self.search_cursor_pos = len(self.search_keyword)  # Set cursor to end of keyword

        return True


def main(stdscr: curses.window):
    curses.curs_set(0)

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        # Initialize color pair 1 for normal text with default foreground and background colors
        curses.init_pair(1, -1, -1)  # -1 means default terminal colors

    view = ImageView(stdscr)
    while True:
        view.display()

        k = stdscr.getch()
        # Clear the screen. This is especially important when resizing the terminal, which send a curses.KEY_RESIZE.
        stdscr.erase()

        if not view.handle_input(k):
            break


if __name__ == "__main__":
    curses.wrapper(main)
