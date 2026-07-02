"""Backend-aware figure display.

Library plotting code should not call ``plt.show()`` unconditionally: on
non-interactive backends (Agg in headless scripts and CI, or the file-output
backends) it emits a UserWarning and does nothing useful, and unclosed
figures accumulate memory across calls. This helper shows the figure when a
display is available and closes it when there is not.

Notebook users lose nothing: inline backends render every figure created in
a cell automatically, so ``plt.show()`` is effectively a flush there.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Backends that render to files or memory rather than a screen. Checked by
# exact name: substring checks would misfire because GUI backends like
# qtagg and tkagg also contain "agg".
_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}


def show_figure() -> None:
    """Show the current figure on interactive backends, close it otherwise.

    Call in place of ``plt.show()`` at the end of a plotting function. On a
    non-interactive backend the figure is closed instead of shown, which
    avoids the FigureCanvasAgg warning and frees the figure's memory in
    long-running processes and test suites.
    """

    backend = matplotlib.get_backend().lower()

    if backend in _NON_INTERACTIVE_BACKENDS:
        plt.close(plt.gcf())
        return

    plt.show()
