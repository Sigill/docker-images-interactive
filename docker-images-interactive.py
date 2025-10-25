#!/usr/bin/env python3
import curses
import curses.ascii
import json
import subprocess
from typing import Any, Iterable, List, Set, Tuple, TypedDict


# pylint: disable=too-few-public-methods
class Config():
    enable_delete_confirmation: bool = True


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
    return run_docker_command(['docker', 'images', '--format', '{{json .}}', '--no-trunc'])


# Utility to run 'docker ps -a --format {{json .}}' and return list of ContainerInfo
def list_docker_containers() -> List[ContainerInfo]:
    return run_docker_command(['docker', 'ps', '-a', '--format', '{{json .}}', '--no-trunc'])


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


def rell(value: str, sz: int):
    return (value[:(sz-1)] + '…') if len(value) > sz else value


def lell(value: str, sz: int):
    return ('…' + value[-(sz-1):]) if len(value) > sz else value


def delete_image(img: ImageInfo):
    if img['Repository'] != '<none>' and img['Tag'] != '<none>':
        name = f"{img['Repository']}:{img['Tag']}"
    else:
        name = img['ID']

    # check=False, because docker rmi might fail if e.g. the image has already been removed from outside this script.
    subprocess.run(['docker', 'rmi', name], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def delete_container(container: ContainerInfo):
    # check=False, because docker rmi might fail if e.g. the image has already been removed from outside this script.
    subprocess.run(['docker', 'rm', container['ID']], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


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


def compute_columns_width(
    headers_width: list[int],
    columns_width: Iterable[List[int | str]]
) -> List[int]:
    max_width = headers_width.copy()

    for column_width in columns_width:
        for i, value in enumerate(column_width):
            sz = len(value) if isinstance(value, str) else value
            if sz > max_width[i]:
                max_width[i] = sz

    return max_width


def format_columns(columns: Iterable[Tuple[str, int]]) -> str:
    return '  '.join(f"{value:{width}}" for (value, width) in columns)


def remove_prefix(value: str, prefix: str) -> str:
    # Starting with python3.9, use str.removeprefix().
    return value[len(prefix):] if value.startswith(prefix) else value


def pretty_id_or_name(value: str, sz: int | None) -> str:
    prefix = 'sha256:'
    if value.startswith(prefix):
        return value[len(prefix):][:sz]
    return value


# pylint: disable=too-many-instance-attributes
class ImageView:
    def __init__(self, stdscr: curses.window, config: Config):
        self.stdscr = stdscr
        self.config = config

        self.image_container_pairs = get_images_with_containers()

        self.current_image = 0
        self.selected_images: Set[int] = set()

        self.confirm_delete_mode = False

        self.max_display_lines = 0
        self.scroll_offset = 0

        self.search_mode = False
        self.search_keyword = ""
        self.saved_search_keyword = ""
        self.search_cursor_pos = 0

    def display(self) -> None:
        max_y, _ = self.stdscr.getmaxyx()
        self.max_display_lines = max_y - 2  # Reserve 2 lines for header and instructions

        # Calculate scroll offset to keep selected item visible
        if self.current_image < self.scroll_offset:
            self.scroll_offset = self.current_image
        elif self.current_image >= self.scroll_offset + self.max_display_lines:
            self.scroll_offset = self.current_image - self.max_display_lines + 1

        # Ensure scroll offset is not negative
        self.scroll_offset = max(0, self.scroll_offset)

        display_pairs = self.image_container_pairs
        # Apply filtering if in search mode
        used_search_keyword = self.search_keyword if self.search_mode else self.saved_search_keyword
        if used_search_keyword:
            display_pairs = filter_images(self.image_container_pairs, used_search_keyword)

        headers = [' ', 'IMAGE ID', 'REPOSITORY', 'TAG', 'SIZE', 'CREATED', 'USED']
        columns_width = compute_columns_width(
            [len(h) for h in headers],
            ([
                1,
                remove_prefix(value=img['ID'], prefix='sha256:')[:12],
                img['Repository'],
                img['Tag'],
                img['Size'],
                img['CreatedSince'],
                1
            ] for (img, _) in display_pairs)
        )

        self.stdscr.addstr(0, 0, format_columns(zip(headers, columns_width)))

        # Display only the visible range of images
        for display_idx, (img, containers) in enumerate(display_pairs[self.scroll_offset:][:self.max_display_lines]):
            idx = display_idx + self.scroll_offset
            line = format_columns(zip(
                [
                    '>' if idx in self.selected_images else ' ',
                    remove_prefix(value=img['ID'], prefix='sha256:')[:12],
                    img['Repository'],
                    img['Tag'],
                    img['Size'],
                    img['CreatedSince'],
                    '*' if len(containers) > 0 else ' '
                ],
                columns_width
            ))
            if idx == self.current_image:
                self.stdscr.addstr(1+display_idx, 0, line, curses.A_REVERSE)
            else:
                self.stdscr.addstr(1+display_idx, 0, line)

        if self.search_mode:
            self.stdscr.addstr(max_y-1, 0, "Search: ")
            search_start_x = 8  # Position after "Search: "
            display_editable_text(self.stdscr, self.search_keyword, self.search_cursor_pos, max_y-1, search_start_x)
        elif self.confirm_delete_mode:
            if len(self.selected_images) > 0:
                self.stdscr.addstr(max_y-1, 0, f"Delete {len(self.selected_images)} images? (y/n) [Y to skip confirmation]")
            else:
                image = display_pairs[self.current_image][0]
                self.stdscr.addstr(max_y-1, 0, f"Delete image {image['Repository']}:{image['Tag']}? (y/n) [Y to skip confirmation]")
        else:
            self.stdscr.addstr(max_y-1, 0, "(q: quit, d: delete, r/F5: refresh, g: first, G: last, /: search, v: container list view)")

    def _delete_selected_images(self) -> None:
        if len(self.selected_images) == 0:
            img, _ = self.image_container_pairs[self.current_image]
            delete_image(img)
        else:
            for idx in self.selected_images:
                img, _ = self.image_container_pairs[idx]
                delete_image(img)

            self.selected_images.clear()

        self.image_container_pairs = get_images_with_containers()
        self.current_image = min(self.current_image, len(self.image_container_pairs)-1)

    # pylint: disable=too-many-branches,too-many-statements
    def handle_input(self, k: int) -> bool:
        if self.search_mode:
            self.current_image = 0

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
            if k in [ord('y'), ord('Y'), curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
                if k == ord('Y'):
                    self.config.enable_delete_confirmation = False

                self._delete_selected_images()
                self.confirm_delete_mode = False
            elif k in [ord('n'), ord('q'), curses.ascii.ESC]:
                self.confirm_delete_mode = False
        else:
            if k == curses.KEY_UP:
                self.current_image = max(0, self.current_image-1)
            elif k == curses.KEY_DOWN:
                self.current_image = min(len(self.image_container_pairs)-1, self.current_image+1)
            elif k == curses.KEY_NPAGE:  # Page Down
                if len(self.image_container_pairs) > 0:
                    # If not at the last visible item, jump to last visible
                    last_visible = min(self.scroll_offset + self.max_display_lines - 1, len(self.image_container_pairs) - 1)
                    if self.current_image != last_visible:
                        self.current_image = last_visible
                    else:
                        self.current_image = min(self.current_image + self.max_display_lines, len(self.image_container_pairs) - 1)
            elif k == curses.KEY_PPAGE:  # Page Up
                if len(self.image_container_pairs) > 0:
                    # If not at the first visible item, jump to first visible
                    first_visible = self.scroll_offset
                    if self.current_image != first_visible:
                        self.current_image = first_visible
                    else:
                        self.current_image = max(self.current_image - self.max_display_lines, 0)
            elif k == ord('g'):  # Select first image
                self.current_image = 0
            elif k == ord('G'):  # Select last image
                self.current_image = len(self.image_container_pairs) - 1
            elif k == ord('d'):
                if self.config.enable_delete_confirmation:
                    self.confirm_delete_mode = True
                else:
                    self._delete_selected_images()
            elif k == ord('q') or k == curses.ascii.ESC:
                return False
            elif k == ord('r') or k == curses.KEY_F5:  # Refresh
                self.image_container_pairs = get_images_with_containers()
                self.selected_images.clear()
                self.current_image = 0
            elif k == ord('/'):
                # Enter search mode
                self.search_mode = True
                self.search_keyword = self.saved_search_keyword  # Use the saved keyword when entering search mode
                self.search_cursor_pos = len(self.search_keyword)  # Set cursor to end of keyword
            elif k == ord(' '):  # Space key
                if self.current_image in self.selected_images:
                    self.selected_images.remove(self.current_image)
                else:
                    self.selected_images.add(self.current_image)

        return True


class ContainerView:
    def __init__(self, stdscr: curses.window, config: Config):
        self.stdscr = stdscr
        self.config = config

        self.containers = list_docker_containers()

        self.current_container = 0
        self.selected_containers: Set[int] = set()

        self.confirm_delete_mode = False

        self.max_display_lines = 0
        self.scroll_offset = 0

    def display(self) -> None:
        max_y, _ = self.stdscr.getmaxyx()
        self.max_display_lines = max_y - 2  # Reserve 2 lines for header and instructions

        if not self.containers:
            self.stdscr.addstr(1, 0, "No containers found.")
            return

        # Calculate scroll offset to keep selected item visible
        if self.current_container < self.scroll_offset:
            self.scroll_offset = self.current_container
        elif self.current_container >= self.scroll_offset + self.max_display_lines:
            self.scroll_offset = self.current_container - self.max_display_lines + 1

        # Ensure scroll offset is not negative
        self.scroll_offset = max(0, self.scroll_offset)

        headers = [' ', 'CONTAINER ID', 'IMAGE', 'COMMAND', 'CREATED', 'STATUS', 'NAMES']
        columns_width = compute_columns_width(
            [len(h) for h in headers],
            ([
                1,
                cont['ID'][:12],
                pretty_id_or_name(cont['Image'], 12),
                cont['Command'],
                cont['RunningFor'],
                cont['Status'],
                cont['Names']
            ] for cont in self.containers)
        )

        self.stdscr.addstr(0, 0, format_columns(zip(headers, columns_width)))

        for display_idx, container in enumerate(self.containers[self.scroll_offset:][:self.max_display_lines]):
            line = format_columns(zip(
                [
                    '>' if display_idx in self.selected_containers else ' ',
                    container['ID'][:12],
                    pretty_id_or_name(container['Image'], 12),
                    container['Command'],
                    container['RunningFor'],
                    container['Status'],
                    container['Names'],
                ],
                columns_width
            ))
            if display_idx + self.scroll_offset == self.current_container:
                self.stdscr.addstr(1+display_idx, 0, line, curses.A_REVERSE)
            else:
                self.stdscr.addstr(1+display_idx, 0, line)

        if self.confirm_delete_mode:
            if len(self.selected_containers) > 0:
                self.stdscr.addstr(max_y-1, 0, f"Delete {len(self.selected_containers)} containers? (y/n) [Y to skip confirmation]")
            else:
                container = self.containers[self.current_container]
                self.stdscr.addstr(max_y-1, 0, f"Delete container {container['ID'][:8]} - {container['Names']}? (y/n) [Y to skip confirmation]")
        else:
            self.stdscr.addstr(max_y-1, 0, "(q: quit, d: delete, r/F5: refresh, g: first, G: last, v: image list view)")

    def _delete_selected_containers(self) -> None:
        if len(self.selected_containers) == 0:
            container = self.containers[self.current_container]
            delete_container(container)
        else:
            for idx in self.selected_containers:
                container = self.containers[idx]
                delete_container(container)

            self.selected_containers.clear()

        self.containers = list_docker_containers()
        self.current_container = min(self.current_container, len(self.containers)-1)

    # pylint: disable=too-many-branches
    def handle_input(self, k: int) -> bool:
        if self.confirm_delete_mode:
            if k in [ord('y'), ord('Y'), curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
                if k == ord('Y'):
                    self.config.enable_delete_confirmation = False

                self._delete_selected_containers()
                self.confirm_delete_mode = False
            elif k in [ord('n'), ord('q'), curses.ascii.ESC]:
                self.confirm_delete_mode = False
        else:
            if k == curses.KEY_UP:
                self.current_container = max(0, self.current_container-1)
            elif k == curses.KEY_DOWN:
                self.current_container = min(len(self.containers)-1, self.current_container+1)
            elif k == curses.KEY_NPAGE:  # Page Down
                if len(self.containers) > 0:
                    last_visible = min(self.scroll_offset + self.max_display_lines - 1, len(self.containers) - 1)
                    if self.current_container != last_visible:
                        self.current_container = last_visible
                    else:
                        self.current_container = min(self.current_container + self.max_display_lines, len(self.containers) - 1)
            elif k == curses.KEY_PPAGE:  # Page Up
                if len(self.containers) > 0:
                    first_visible = self.scroll_offset
                    if self.current_container != first_visible:
                        self.current_container = first_visible
                    else:
                        self.current_container = max(self.current_container - self.max_display_lines, 0)
            elif k == ord('g'):  # Select first container
                self.current_container = 0
            elif k == ord('G'):  # Select last container
                self.current_container = len(self.containers) - 1
            elif k == ord('d'):
                if self.config.enable_delete_confirmation:
                    self.confirm_delete_mode = True
                else:
                    self._delete_selected_containers()
            elif k == ord('q') or k == curses.ascii.ESC:
                return False
            elif k == ord('r') or k == curses.KEY_F5:  # Refresh
                self.containers = list_docker_containers()
                self.selected_containers.clear()
                self.current_container = 0
            elif k == ord('/'):
                # Enter search mode (not implemented here)
                pass
            elif k == ord(' '):  # Space key
                if self.current_container in self.selected_containers:
                    self.selected_containers.remove(self.current_container)
                else:
                    self.selected_containers.add(self.current_container)

        return True


def main_curses(stdscr: curses.window, config: Config):
    curses.curs_set(0)

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        # Initialize color pair 1 for normal text with default foreground and background colors
        curses.init_pair(1, -1, -1)  # -1 means default terminal colors

    view: ImageView | ContainerView = ImageView(stdscr, config)

    while True:
        view.display()
        stdscr.refresh()

        k = stdscr.getch()
        # Clear the screen. This is especially important when resizing the terminal, which send a curses.KEY_RESIZE.
        stdscr.erase()

        if k == ord('v'):  # Switch view
            view = ContainerView(stdscr, config) if isinstance(view, ImageView) else ImageView(stdscr, config)
            continue

        if not view.handle_input(k):
            break


def main():
    config = Config()
    curses.wrapper(lambda w: main_curses(w, config))


if __name__ == "__main__":
    main()
