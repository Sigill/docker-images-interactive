#!/usr/bin/env python3
import curses
import curses.ascii
import json
import subprocess
from typing import Any, Callable, Iterable, List, Set, Tuple, TypedDict


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


def filter_images(
    image_pairs: List[Tuple[ImageInfo, List[ContainerInfo]]],
    keyword: str
) -> List[Tuple[ImageInfo, List[ContainerInfo]]]:
    """Filter image with keyword in repository:tag"""
    return [
        (img, containers)
        for (img, containers) in image_pairs
        if keyword.lower() in f"{img['Repository']}:{img['Tag']}".lower()
    ]


def filter_containers(
    containers: List[ContainerInfo],
    keyword: str
) -> List[ContainerInfo]:
    """Filter containers with keyword in repository:tag"""
    return [container for container in containers if keyword.lower() in container['Image']]


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


class Filter:
    enabled = False
    search_keyword = ""
    saved_search_keyword = ""
    search_cursor_pos = 0

    def get_effective_keyword(self):
        return self.search_keyword if self.enabled else self.saved_search_keyword

    def enable(self):
        self.enabled = True
        self.search_keyword = self.saved_search_keyword  # Use the saved keyword when entering search mode
        self.search_cursor_pos = len(self.search_keyword)  # Set cursor to end of keyword

    def handle_input(self, k: int):
        if k in [curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
            # Validate and save the search keyword
            self.saved_search_keyword = self.search_keyword
            self.enabled = False
            self.search_cursor_pos = 0
        elif k == curses.ascii.ESC:  # ESC key
            # Cancel search mode
            self.enabled = False
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


# pylint: disable=too-many-instance-attributes
class ListController:
    def __init__(
        self,
        get_list_length: Callable[[], int],
        on_refresh: Callable[[], None],
        on_delete: Callable[[], None],
    ) -> None:
        self.filter = Filter()

        self.current: int = 0
        self.selected_items: Set[int] = set()

        self.max_display_lines = 0
        self.scroll_offset = 0

        self.get_list_length = get_list_length
        self.on_refresh = on_refresh
        self.on_delete = on_delete

    # pylint: disable=too-many-branches,too-many-return-statements
    def handle_input(self, k: int):
        """
        Return:
         - None if the event was not handled by this function.
         - True if the event was handled and the event loop shall continue.
         - False if the event was handled and the app shall exit.
        """
        if k == curses.KEY_UP:
            self.current = max(0, self.current-1)
            return True
        elif k == curses.KEY_DOWN:
            self.current = min(self.get_list_length()-1, self.current+1)
            return True
        elif k == curses.KEY_NPAGE:  # Page Down
            if self.get_list_length() > 0:
                last_visible = min(self.scroll_offset + self.max_display_lines - 1, self.get_list_length() - 1)
                if self.current != last_visible:
                    self.current = last_visible
                else:
                    self.current = min(self.current + self.max_display_lines, self.get_list_length() - 1)
            return True
        elif k == curses.KEY_PPAGE:  # Page Up
            if self.get_list_length() > 0:
                first_visible = self.scroll_offset
                if self.current != first_visible:
                    self.current = first_visible
                else:
                    self.current = max(self.current - self.max_display_lines, 0)
            return True
        elif k == ord('g'):  # Select first container
            self.current = 0
            return True
        elif k == ord('G'):  # Select last container
            self.current = self.get_list_length() - 1
            return True
        elif k == ord('d'):
            self.on_delete()
            return True
        elif k == ord('q') or k == curses.ascii.ESC:
            return False
        elif k == ord('r') or k == curses.KEY_F5:  # Refresh
            self.selected_items.clear()
            self.current = 0

            self.on_refresh()
            return True
        elif k == ord('/'):
            self.filter.enable()
            return True
        elif k == ord(' '):  # Space key
            if self.current in self.selected_items:
                self.selected_items.remove(self.current)
            else:
                self.selected_items.add(self.current)
            return True

        return None


# pylint: disable=too-many-instance-attributes
class ImageView:
    def __init__(self, stdscr: curses.window, config: Config):
        self.stdscr = stdscr
        self.config = config

        self.image_container_pairs = get_images_with_containers()

        self.list_controller = ListController(
            get_list_length=lambda: len(self.image_container_pairs),
            on_refresh=self._refresh,
            on_delete=self._on_delete
        )

        self.filter = self.list_controller.filter

        self.confirm_delete_mode = False

    def _refresh(self):
        self.image_container_pairs = get_images_with_containers()

    def _on_delete(self):
        if self.config.enable_delete_confirmation:
            self.confirm_delete_mode = True
        else:
            self._delete_selected_images()

    def _delete_selected_images(self) -> None:
        if len(self.list_controller.selected_items) == 0:
            img, _ = self.image_container_pairs[self.list_controller.current]
            delete_image(img)
        else:
            for idx in self.list_controller.selected_items:
                img, _ = self.image_container_pairs[idx]
                delete_image(img)

            self.list_controller.selected_items.clear()

        self._refresh()
        self.list_controller.current = min(self.list_controller.current, len(self.image_container_pairs)-1)

    def display(self) -> None:
        max_y, _ = self.stdscr.getmaxyx()
        self.list_controller.max_display_lines = max_y - 2  # Reserve 2 lines for header and instructions

        # Calculate scroll offset to keep selected item visible
        if self.list_controller.current < self.list_controller.scroll_offset:
            self.list_controller.scroll_offset = self.list_controller.current
        elif self.list_controller.current >= self.list_controller.scroll_offset + self.list_controller.max_display_lines:
            self.list_controller.scroll_offset = self.list_controller.current - self.list_controller.max_display_lines + 1

        # Ensure scroll offset is not negative
        self.list_controller.scroll_offset = max(0, self.list_controller.scroll_offset)

        display_pairs = filter_images(self.image_container_pairs, self.filter.get_effective_keyword())

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
        for display_idx, (img, containers) in enumerate(display_pairs[self.list_controller.scroll_offset:][:self.list_controller.max_display_lines]):
            idx = display_idx + self.list_controller.scroll_offset
            line = format_columns(zip(
                [
                    '>' if idx in self.list_controller.selected_items else ' ',
                    remove_prefix(value=img['ID'], prefix='sha256:')[:12],
                    img['Repository'],
                    img['Tag'],
                    img['Size'],
                    img['CreatedSince'],
                    '*' if len(containers) > 0 else ' '
                ],
                columns_width
            ))
            if idx == self.list_controller.current:
                self.stdscr.addstr(1+display_idx, 0, line, curses.A_REVERSE)
            else:
                self.stdscr.addstr(1+display_idx, 0, line)

        if self.filter.enabled:
            self.stdscr.addstr(max_y-1, 0, "Search: ")
            search_start_x = 8  # Position after "Search: "
            display_editable_text(self.stdscr, self.filter.search_keyword, self.filter.search_cursor_pos, max_y-1, search_start_x)
        elif self.confirm_delete_mode:
            if len(self.list_controller.selected_items) > 0:
                self.stdscr.addstr(max_y-1, 0, f"Delete {len(self.list_controller.selected_items)} images? (y/n) [Y to skip confirmation]")
            else:
                image = display_pairs[self.list_controller.current][0]
                self.stdscr.addstr(max_y-1, 0, f"Delete image {image['Repository']}:{image['Tag']}? (y/n) [Y to skip confirmation]")
        else:
            self.stdscr.addstr(max_y-1, 0, "(q: quit, d: delete, r/F5: refresh, g: first, G: last, /: search, v: container list view)")

    def handle_input(self, k: int) -> bool:
        if self.filter.enabled:
            self.list_controller.current = 0
            self.filter.handle_input(k)
        elif self.confirm_delete_mode:
            if k in [ord('y'), ord('Y'), curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
                if k == ord('Y'):
                    self.config.enable_delete_confirmation = False

                self._delete_selected_images()
                self.confirm_delete_mode = False
            elif k in [ord('n'), ord('q'), curses.ascii.ESC]:
                self.confirm_delete_mode = False
        else:
            status = self.list_controller.handle_input(k)
            return status if status is not None else True

        return True


class ContainerView:
    def __init__(self, stdscr: curses.window, config: Config):
        self.stdscr = stdscr
        self.config = config

        self.containers = list_docker_containers()

        self.list_controller = ListController(
            get_list_length=lambda: len(self.containers),
            on_refresh=self._refresh,
            on_delete=self._on_delete
        )

        self.filter = self.list_controller.filter

        self.confirm_delete_mode = False

    def _refresh(self):
        self.containers = list_docker_containers()

    def _on_delete(self):
        if self.config.enable_delete_confirmation:
            self.confirm_delete_mode = True
        else:
            self._delete_selected_containers()

    def _delete_selected_containers(self) -> None:
        if len(self.list_controller.selected_items) == 0:
            container = self.containers[self.list_controller.current]
            delete_container(container)
        else:
            for idx in self.list_controller.selected_items:
                container = self.containers[idx]
                delete_container(container)

            self.list_controller.selected_items.clear()

        self._refresh()
        self.list_controller.current = min(self.list_controller.current, len(self.containers)-1)

    def display(self) -> None:
        max_y, _ = self.stdscr.getmaxyx()
        self.list_controller.max_display_lines = max_y - 2  # Reserve 2 lines for header and instructions

        if not self.containers:
            self.stdscr.addstr(1, 0, "No containers found.")
            return

        # Calculate scroll offset to keep selected item visible
        if self.list_controller.current < self.list_controller.scroll_offset:
            self.list_controller.scroll_offset = self.list_controller.current
        elif self.list_controller.current >= self.list_controller.scroll_offset + self.list_controller.max_display_lines:
            self.list_controller.scroll_offset = self.list_controller.current - self.list_controller.max_display_lines + 1

        # Ensure scroll offset is not negative
        self.list_controller.scroll_offset = max(0, self.list_controller.scroll_offset)

        display_containers = filter_containers(self.containers, self.filter.get_effective_keyword())

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
            ] for cont in display_containers)
        )

        self.stdscr.addstr(0, 0, format_columns(zip(headers, columns_width)))

        for display_idx, container in enumerate(display_containers[self.list_controller.scroll_offset:][:self.list_controller.max_display_lines]):
            idx = display_idx + self.list_controller.scroll_offset
            line = format_columns(zip(
                [
                    '>' if display_idx in self.list_controller.selected_items else ' ',
                    container['ID'][:12],
                    pretty_id_or_name(container['Image'], 12),
                    container['Command'],
                    container['RunningFor'],
                    container['Status'],
                    container['Names'],
                ],
                columns_width
            ))
            if idx == self.list_controller.current:
                self.stdscr.addstr(1+display_idx, 0, line, curses.A_REVERSE)
            else:
                self.stdscr.addstr(1+display_idx, 0, line)

        if self.filter.enabled:
            self.stdscr.addstr(max_y-1, 0, "Search: ")
            search_start_x = 8  # Position after "Search: "
            display_editable_text(self.stdscr, self.filter.search_keyword, self.filter.search_cursor_pos, max_y-1, search_start_x)
        elif self.confirm_delete_mode:
            if len(self.list_controller.selected_items) > 0:
                self.stdscr.addstr(max_y-1, 0, f"Delete {len(self.list_controller.selected_items)} containers? (y/n) [Y to skip confirmation]")
            else:
                container = self.containers[self.list_controller.current]
                self.stdscr.addstr(max_y-1, 0, f"Delete container {container['ID'][:8]} - {container['Names']}? (y/n) [Y to skip confirmation]")
        else:
            self.stdscr.addstr(max_y-1, 0, "(q: quit, d: delete, r/F5: refresh, g: first, G: last, v: image list view)")

    def handle_input(self, k: int) -> bool:
        if self.filter.enabled:
            self.list_controller.current = 0
            self.filter.handle_input(k)
        elif self.confirm_delete_mode:
            if k in [ord('y'), ord('Y'), curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
                if k == ord('Y'):
                    self.config.enable_delete_confirmation = False

                self._delete_selected_containers()
                self.confirm_delete_mode = False
            elif k in [ord('n'), ord('q'), curses.ascii.ESC]:
                self.confirm_delete_mode = False
        else:
            status = self.list_controller.handle_input(k)
            return status if status is not None else True

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
