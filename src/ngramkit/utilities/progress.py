# utils/progress.py
"""Common progress display and banner formatting utilities."""

from typing import Dict, Any, Optional

__all__ = ["ProgressDisplay"]


class ProgressDisplay:
    """Utility class for consistent progress banners and section headers."""

    def __init__(self, width: int = 100):
        """Initialize with display width.

        Args:
            width: Character width for banners and separators
        """
        self.width = width

    def print_banner(
        self, title: str, style: str = "═", include_blank: bool = False
    ) -> None:
        """Print a major section banner.

        Args:
            title: Banner title
            style: Separator character (═ for major sections)
            include_blank: Add blank line before banner
        """
        if include_blank:
            print()
        print(title)
        print(style * self.width)

    def print_section(self, title: str, style: str = "—") -> None:
        """Print a subsection header.

        Args:
            title: Section title
            style: Separator character (— for subsections)
        """
        print()
        print(title)
        print(style * self.width)

    def print_config_items(self, items: Dict[str, Any], indent: str = "") -> None:
        """Print configuration items in aligned format.

        Args:
            items: Dictionary of config keys and values
            indent: Optional indentation prefix
        """
        if not items:
            return

        # Find max key length for alignment
        max_key_len = max(len(str(k)) for k in items.keys())

        for key, value in items.items():
            padding = " " * (max_key_len - len(str(key)))
            print(f"{indent}{key}:{padding} {value}")

    def print_summary_box(
        self,
        title: str,
        items: Dict[str, Any],
        box_width: Optional[int] = None,
    ) -> None:
        """Print a boxed summary.

        Args:
            title: Box title
            items: Dictionary of summary items
            box_width: Width of box (defaults to self.width)
        """
        if box_width is None:
            box_width = self.width

        # Calculate maximum content width needed
        title_width = len(f" {title} ")
        item_widths = [len(f" {key}: {value} ") for key, value in items.items()]
        max_content_width = max([title_width] + item_widths)

        # Use the larger of requested box_width or content width
        actual_width = max(box_width, max_content_width + 2)

        # Top border
        print("┌" + "─" * (actual_width - 2) + "┐")

        # Title
        title_text = f" {title} "
        padding = " " * (actual_width - len(title_text) - 2)
        print("│" + title_text + padding + "│")

        # Separator
        print("├" + "─" * (actual_width - 2) + "┤")

        # Items
        for key, value in items.items():
            item_text = f" {key}: {value} "
            padding = " " * (actual_width - len(item_text) - 2)
            print("│" + item_text + padding + "│")

        # Bottom border
        print("└" + "─" * (actual_width - 2) + "┘")

    @staticmethod
    def format_rate(count: int, elapsed_seconds: float, unit: str = "items") -> str:
        """Format a processing rate.

        Args:
            count: Number of items processed
            elapsed_seconds: Time elapsed
            unit: Unit name (e.g., "items", "records", "keys")

        Returns:
            Formatted rate string

        Examples:
            >>> ProgressDisplay.format_rate(10000, 2.5, "records")
            '4,000 records/sec'
        """
        if elapsed_seconds <= 0:
            return f"0 {unit}/sec"

        rate = count / elapsed_seconds

        if rate >= 1000:
            return f"{rate:,.0f} {unit}/sec"
        elif rate >= 10:
            return f"{rate:.1f} {unit}/sec"
        else:
            return f"{rate:.2f} {unit}/sec"

    @staticmethod
    def format_percentage(numerator: int, denominator: int) -> str:
        """Format a percentage.

        Args:
            numerator: Numerator value
            denominator: Denominator value

        Returns:
            Formatted percentage string

        Examples:
            >>> ProgressDisplay.format_percentage(75, 100)
            '75.0%'
        """
        if denominator == 0:
            return "0.0%"
        return f"{(numerator / denominator) * 100:.1f}%"
