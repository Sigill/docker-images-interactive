#!/usr/bin/env python3
import curses
import curses.ascii
import json
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, List, Sequence, Set, Tuple, TypedDict, Union


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
) -> List[int]:
    """Filter image with keyword in repository:tag"""
    return [
        i
        for i, (img, _) in enumerate(image_pairs)
        if keyword.lower() in f"{img['Repository']}:{img['Tag']}".lower()
    ]


def filter_containers(
    containers: List[ContainerInfo],
    keyword: str
) -> List[int]:
    """Filter containers with keyword in repository:tag"""
    return [
        i
        for i, container in enumerate(containers)
        if keyword.lower() in container['Image'].lower()
    ]


def display_editable_text(stdscr: curses.window, text: str, cursor_position: int, y: int, x: int):
    # Display the search keyword with cursor highlighting
    for i, char in enumerate(text):
        # Highlight the character at cursor position
        style = curses.A_REVERSE if i == cursor_position else curses.A_NORMAL
        stdscr.addch(y, x + i, char, style)

    # If cursor is at the end, we need to add a reverse space to indicate cursor position
    if cursor_position == len(text):
        stdscr.addch(y, x + cursor_position, ' ', curses.A_REVERSE)


def compute_columns_width(
    headers_width: List[int],
    columns_width: Iterable[List[str]]
) -> List[int]:
    max_width = headers_width.copy()

    for column_width in columns_width:
        for i, value in enumerate(column_width):
            max_width[i] = max(max_width[i], len(value))

    return max_width


def format_columns(columns: Iterable[Tuple[str, int]]) -> str:
    return '  '.join(f"{value:{width}}" for (value, width) in columns)


def remove_prefix(value: str, prefix: str) -> str:
    # Starting with python3.9, use str.removeprefix().
    return value[len(prefix):] if value.startswith(prefix) else value


def pretty_id_or_name(value: str, sz: Union[int, None]) -> str:
    prefix = 'sha256:'
    if value.startswith(prefix):
        return value[len(prefix):][:sz]
    return value


class ScrollController:
    def __init__(self) -> None:
        self.offset: int = 0
        self.current: int = 0
        self.available_size: int = 0

    def adjust_offset(self, available_size: int):
        self.available_size = available_size

        # Calculate scroll offset to keep selected item visible
        if self.current < self.offset:
            self.offset = self.current
        elif self.current >= self.offset + available_size:
            self.offset = self.current - available_size + 1

        # Ensure scroll offset is not negative
        self.offset = max(0, self.offset)

    def prev(self):
        self.current = max(0, self.current-1)

    def next(self, collection_size: int):
        self.current = min(self.current+1, collection_size-1)

    def first(self):
        self.current = 0

    def last(self, collection_size: int):
        self.current = collection_size - 1

    def prev_page(self):
        first_visible = self.offset
        if self.current != first_visible:
            self.current = first_visible
        else:
            self.current = max(self.current - self.available_size, 0)

    def next_page(self, collection_size: int):
        last_visible = min(self.offset + self.available_size - 1, collection_size - 1)
        if self.current != last_visible:
            self.current = last_visible
        else:
            self.current = min(self.current + self.available_size, collection_size - 1)


class Filter:
    def __init__(self, on_change: Callable[[], None]) -> None:
        self.on_change = on_change
        self.editing = False
        self.live_keyword = ""
        self.saved_keyword = ""
        self.cursor_pos = 0

    def get_effective_keyword(self):
        return self.live_keyword if self.editing else self.saved_keyword

    def enable_edit(self):
        self.editing = True
        self.live_keyword = self.saved_keyword  # Use the saved keyword when entering search mode
        self.cursor_pos = len(self.live_keyword)  # Set cursor to end of keyword

    def handle_input(self, k: int):
        if k in [curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
            # Validate and save the search keyword
            self.saved_keyword = self.live_keyword
            self.editing = False
            self.cursor_pos = 0
        elif k == curses.ascii.ESC:  # ESC key
            # Cancel search mode
            self.editing = False
            self.live_keyword = ""
            self.cursor_pos = 0
            self.on_change()
        elif k in [curses.KEY_BACKSPACE, curses.ascii.DEL, curses.ascii.BS]:
            # Backspace
            if self.cursor_pos > 0:
                self.live_keyword = self.live_keyword[:self.cursor_pos-1] + self.live_keyword[self.cursor_pos:]
                self.cursor_pos -= 1
                self.on_change()
        elif k == curses.KEY_LEFT:
            # Left arrow - move cursor left
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
        elif k == curses.KEY_RIGHT:
            # Right arrow - move cursor right
            if self.cursor_pos < len(self.live_keyword):
                self.cursor_pos += 1
        elif k == curses.KEY_DC:  # Delete key
            # Delete character at cursor position
            if self.cursor_pos < len(self.live_keyword):
                self.live_keyword = self.live_keyword[:self.cursor_pos] + self.live_keyword[self.cursor_pos+1:]
                self.on_change()
        elif 32 <= k <= 126:  # Printable characters
            # Add character to search keyword at cursor position
            self.live_keyword = self.live_keyword[:self.cursor_pos] + chr(k) + self.live_keyword[self.cursor_pos:]
            self.cursor_pos += 1
            self.on_change()


class ListView:
    def __init__(self, filter: Filter) -> None:
        self.filter = filter

        self.items: List[int] = []
        self.selected_items: Set[int] = set()

        self.__scroll = ScrollController()

    def set_items(self, items: List[int]):
        self.items = items
        self.__scroll.first()  # TODO try to maintain current item selected.

    def remove_item(self, item: int):
        self.selected_items = set(i - 1 if i > item else i for i in self.selected_items if i != item)
        self.items = [i - 1 if i > item else i for i in self.items if i != item]
        if item <= self.get_current_item():
            self.__scroll.prev()

    def get_selection(self) -> List[int]:
        if len(self.selected_items) > 0:
            return list(self.selected_items)
        else:
            return [self.get_current_item()]

    def get_current_item(self) -> int:
        return self.items[self.__scroll.current]

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def display(
        self,
        stdscr: curses.window,
        headers: List[str],
        data: List[Tuple[int, List[str]]],
        confirm_delete_mode: bool,
        empty_msg: str,
        delete_msg: Callable[[List[int]], str],
        additional_shortcuts: str
    ) -> None:
        max_y, _ = stdscr.getmaxyx()
        self.__scroll.adjust_offset(available_size=max_y-2)  # Reserve 2 lines for header and instructions

        if not data:
            stdscr.addstr(0, 0, empty_msg)
        else:
            columns_width = compute_columns_width([len(h) for h in headers], (columns for (_, columns) in data))

            stdscr.addstr(0, 0, format_columns(zip(headers, columns_width)))

            # Display only the visible range of images
            for line_idx, (idx, columns) in enumerate(
                data[self.__scroll.offset:][:self.__scroll.available_size]
            ):
                line = format_columns(zip(columns, columns_width))
                style = curses.A_REVERSE if idx == self.get_current_item() else curses.A_NORMAL
                stdscr.addstr(1+line_idx, 0, line, style)

        if self.filter.editing:
            stdscr.addstr(max_y-1, 0, "Search: ")
            display_editable_text(
                stdscr,
                self.filter.live_keyword,
                self.filter.cursor_pos,
                max_y-1,
                8  # Position after "Search: "
            )
        elif confirm_delete_mode:
            stdscr.addstr(
                max_y-1, 0,
                f"{delete_msg(self.get_selection())} (y/n) [Y to skip confirmation]"
            )
        else:
            base_shortcuts = "q: quit, d: delete, r/F5: refresh, g: first, G: last, /: search"
            stdscr.addstr(max_y-1, 0, f"{base_shortcuts}, {additional_shortcuts}")

    # pylint: disable=too-many-branches,too-many-return-statements
    def handle_input(self, k: int, on_refresh: Callable[[], None], on_delete: Callable[[], None],):
        """
        Return:
         - None if the event was not handled by this function.
         - True if the event was handled and the event loop shall continue.
         - False if the event was handled and the app shall exit.
        """
        if k == curses.KEY_UP:
            self.__scroll.prev()
            return True
        elif k == curses.KEY_DOWN:
            self.__scroll.next(len(self.items))
            return True
        elif k == curses.KEY_NPAGE:  # Page Down
            self.__scroll.next_page(len(self.items))
            return True
        elif k == curses.KEY_PPAGE:  # Page Up
            self.__scroll.prev_page()
            return True
        elif k == ord('g'):  # Select first container
            self.__scroll.first()
            return True
        elif k == ord('G'):  # Select last container
            self.__scroll.last(len(self.items))
            return True
        elif k == ord('d'):
            on_delete()
            return True
        elif k == ord('q') or k == curses.ascii.ESC:
            return False
        elif k == ord('r') or k == curses.KEY_F5:  # Refresh
            self.selected_items.clear()
            self.__scroll.first()

            on_refresh()
            return True
        elif k == ord('/'):
            self.filter.enable_edit()
            return True
        elif k == ord(' '):  # Space key
            current_item = self.get_current_item()
            if current_item in self.selected_items:
                self.selected_items.remove(current_item)
            else:
                self.selected_items.add(current_item)
            return True

        return None


class ImageView:
    HEADERS: Final[List[str]] = [' ', 'IMAGE ID', 'REPOSITORY', 'TAG', 'SIZE', 'CREATED', 'USED']

    def __init__(self, stdscr: curses.window, config: Config):
        self.stdscr = stdscr
        self.config = config

        self.__refresh()

        self.list_view = ListView(Filter(on_change=self.__apply_filter))

        self.__apply_filter()

        self.confirm_delete_mode = False

    def __refresh(self):
        self.images = get_images_with_containers()

    def __apply_filter(self):
        self.list_view.set_items(
            filter_images(
                self.images,
                self.list_view.filter.get_effective_keyword()
            )
        )

    def __on_delete(self):
        if self.config.enable_delete_confirmation:
            self.confirm_delete_mode = True
        else:
            self.__delete_selected_images()

    def __delete_selected_images(self) -> None:
        for idx in sorted(self.list_view.get_selection(), reverse=True):
            img, _ = self.images[idx]
            delete_image(img)
            del self.images[idx]
            self.list_view.remove_item(idx)

    def __columns(self, idx: int, img: ImageInfo, containers: Sequence[ContainerInfo]):
        return [
            '>' if idx in self.list_view.selected_items else ' ',
            remove_prefix(value=img['ID'], prefix='sha256:')[:12],
            img['Repository'],
            img['Tag'],
            img['Size'],
            img['CreatedSince'],
            '*' if len(containers) > 0 else ' '
        ]

    def display(self) -> None:
        data = [
            (i, self.__columns(i, self.images[i][0], self.images[i][1]))
            for i in self.list_view.items
        ]

        def delete_msg(sel: List[int]):
            if len(sel) == 1:
                image, _ = self.images[sel[0]]
                return f"Delete image {image['Repository']}:{image['Tag']}?"
            else:
                return f"Delete {len(self.list_view.selected_items)} images?"

        self.list_view.display(
            self.stdscr,
            self.HEADERS, data,
            self.confirm_delete_mode,
            "No image found.",
            delete_msg,
            "v: containers view"
        )

    def handle_input(self, k: int) -> bool:
        if self.list_view.filter.editing:
            self.list_view.filter.handle_input(k)
        elif self.confirm_delete_mode:
            if k in [ord('y'), ord('Y'), curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
                if k == ord('Y'):
                    self.config.enable_delete_confirmation = False

                self.__delete_selected_images()
                self.confirm_delete_mode = False
            elif k in [ord('n'), ord('q'), curses.ascii.ESC]:
                self.confirm_delete_mode = False
        else:
            status = self.list_view.handle_input(k, self.__refresh, self.__on_delete)
            return status if status is not None else True

        return True


class ContainerView:
    HEADERS: Final[List[str]] = [' ', 'CONTAINER ID', 'IMAGE', 'COMMAND', 'CREATED', 'STATUS', 'NAMES']

    def __init__(self, stdscr: curses.window, config: Config):
        self.stdscr = stdscr
        self.config = config

        self.__refresh()

        self.list_view = ListView(Filter(on_change=self.__apply_filter))

        self.__apply_filter()

        self.confirm_delete_mode = False

    def __refresh(self):
        self.containers = list_docker_containers()

    def __apply_filter(self):
        self.list_view.set_items(
            filter_containers(
                self.containers,
                self.list_view.filter.get_effective_keyword()
            )
        )

    def __on_delete(self):
        if self.config.enable_delete_confirmation:
            self.confirm_delete_mode = True
        else:
            self.__delete_selected_containers()

    def __delete_selected_containers(self) -> None:
        for idx in sorted(self.list_view.get_selection(), reverse=True):
            container = self.containers[idx]
            delete_container(container)
            del self.containers[idx]
            self.list_view.remove_item(idx)

    def __columns(self, idx: int, container: ContainerInfo):
        return [
            '>' if idx in self.list_view.selected_items else ' ',
            container['ID'][:12],
            pretty_id_or_name(container['Image'], 12),
            container['Command'],
            container['RunningFor'],
            container['Status'],
            container['Names'],
        ]

    def display(self) -> None:
        data = [
            (i, self.__columns(i, self.containers[i]))
            for i in self.list_view.items
        ]

        def delete_msg(sel: List[int]):
            if len(sel) == 1:
                container = self.containers[sel[0]]
                return f"Delete container {container['ID'][:8]} - {container['Names']}?"
            else:
                return f"Delete {len(self.list_view.selected_items)} containers?"

        self.list_view.display(
            self.stdscr,
            self.HEADERS, data,
            self.confirm_delete_mode,
            "No container found.",
            delete_msg,
            "v: images view"
        )

    def handle_input(self, k: int) -> bool:
        if self.list_view.filter.editing:
            self.list_view.filter.handle_input(k)
        elif self.confirm_delete_mode:
            if k in [ord('y'), ord('Y'), curses.KEY_ENTER, curses.ascii.CR, curses.ascii.LF]:
                if k == ord('Y'):
                    self.config.enable_delete_confirmation = False

                self.__delete_selected_containers()
                self.confirm_delete_mode = False
            elif k in [ord('n'), ord('q'), curses.ascii.ESC]:
                self.confirm_delete_mode = False
        else:
            status = self.list_view.handle_input(k, self.__refresh, self.__on_delete)
            return status if status is not None else True

        return True


def get_key_press(stdscr: curses.window) -> int:
    while True:
        key = stdscr.getch()

        if key != curses.ascii.ESC:
            return key

        # Check if there are more characters available (Alt+key combinations)
        stdscr.nodelay(True)
        next_key = stdscr.getch()
        stdscr.nodelay(False)

        if next_key in (curses.ascii.ESC, curses.ERR):
            return key


def main_curses(stdscr: curses.window, config: Config):
    curses.curs_set(0)

    if sys.version_info >= (3, 9):
        # Alt+key combinations are sent as escape sequences.
        # A default 1s delay ESC is applied before reporting an ESC key press to help identifying escape sequences.
        # As this script doesn't use escape sequences, reduce this delay for faster ESC key press interpretation.
        curses.set_escdelay(1)

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        # Initialize color pair 1 for normal text with default foreground and background colors
        curses.init_pair(1, -1, -1)  # -1 means default terminal colors

    view: ImageView | ContainerView = ImageView(stdscr, config)

    while True:
        view.display()
        stdscr.refresh()

        k = get_key_press(stdscr)
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
