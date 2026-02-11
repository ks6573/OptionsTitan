"""
OptionsTitan UI Utilities
DPI-aware sizing and responsive layout helpers
"""

from PySide6.QtWidgets import QApplication


def _safe_screen():
    """Get primary screen; return None if app not running."""
    app = QApplication.instance()
    if app and app.primaryScreen():
        return app.primaryScreen()
    return None


def get_dpi_scale_factor() -> float:
    """Get DPI scaling factor (1.0 = 96 DPI standard)."""
    screen = _safe_screen()
    if screen:
        return screen.logicalDotsPerInch() / 96.0
    return 1.0


def get_screen_size():
    """Return (width, height) of primary screen. Fallback (1920, 1080)."""
    screen = _safe_screen()
    if screen:
        geom = screen.availableGeometry()
        return geom.width(), geom.height()
    return 1920, 1080


def get_scaled_pixel_size(base_size: int) -> int:
    """Scale pixel size by DPI."""
    return max(1, int(base_size * get_dpi_scale_factor()))


def get_screen_percentage_height(percentage: float) -> int:
    """Height as percentage of available screen height."""
    _, height = get_screen_size()
    return max(1, int(height * percentage))


def get_screen_percentage_width(percentage: float) -> int:
    """Width as percentage of available screen width."""
    width, _ = get_screen_size()
    return max(1, int(width * percentage))


def get_responsive_font_size(role: str) -> int:
    """Pixel font size for text roles (scaled by DPI + screen)."""
    scale = get_dpi_scale_factor()
    width, _ = get_screen_size()
    # Adjust base for screen size
    if width < 1440:
        base = 9
    elif width < 1920:
        base = 10
    else:
        base = 11
    base = int(base * scale)
    sizes = {
        'title': int(base * 2.5),
        'heading': int(base * 1.8),
        'subheading': int(base * 1.4),
        'body': base,
        'small': int(base * 0.85),
    }
    return max(8, sizes.get(role, base))


def get_layout_mode() -> str:
    """'compact', 'normal', or 'spacious' based on screen width."""
    width, _ = get_screen_size()
    if width < 1440:
        return 'compact'
    if width < 1920:
        return 'normal'
    return 'spacious'


def get_responsive_spacing() -> int:
    """Base spacing in pixels for layout."""
    _, height = get_screen_size()
    return max(10, min(24, int(height * 0.015)))


def get_sidebar_width() -> int:
    """Responsive sidebar width."""
    width, _ = get_screen_size()
    if width < 1440:
        return 320
    if width < 1920:
        return 380
    return 450
