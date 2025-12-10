"""
Consistent plotting style for SmallML framework figures.

Ensures all figures have uniform appearance for publication.

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, Optional


class SmallMLPlotStyle:
    """
    Consistent figure styling across the entire paper.

    Provides standardized colors, sizes, and formatting to ensure
    all framework figures have uniform professional appearance.

    Attributes
    ----------
    FIGSIZE_MAIN : Tuple[int, int]
        Default figure size for main plots (10, 6)
    FIGSIZE_SECONDARY : Tuple[int, int]
        Figure size for secondary plots (8, 6)
    FIGSIZE_SQUARE : Tuple[int, int]
        Figure size for square plots (8, 8)
    DPI : int
        Resolution for saved figures (300 for publication)
    COLORS : Dict[str, str]
        Standard color palette used in the entire paper

    Examples
    --------
    >>> # Apply style to all plots
    >>> SmallMLPlotStyle.apply()
    >>>
    >>> # Create figure with standard size
    >>> fig, ax = SmallMLPlotStyle.create_figure()
    >>>
    >>> # Use standard colors
    >>> ax.plot(x, y, color=SmallMLPlotStyle.COLORS['primary'])
    """

    # Standard figure sizes
    FIGSIZE_MAIN = (10, 6)
    FIGSIZE_SECONDARY = (8, 6)
    FIGSIZE_SQUARE = (8, 8)
    FIGSIZE_WIDE = (12, 5)

    # Publication quality DPI
    DPI = 300

    # Standard color palette (Tab10 colormap colors)
    COLORS = {
        'primary': '#1f77b4',      # Tab blue
        'secondary': '#ff7f0e',    # Tab orange
        'success': '#2ca02c',      # Tab green
        'danger': '#d62728',       # Tab red
        'warning': '#ffbb78',      # Light orange
        'info': '#aec7e8',         # Light blue
        'mle': '#2ca02c',          # Green for MLE
        'hierarchical': '#1f77b4', # Blue for hierarchical
        'population': '#d62728',   # Red for population mean
        'prior': '#ff7f0e',        # Orange for priors
        'catboost': '#1f77b4',     # Blue for trained model
        'random': '#7f7f7f',       # Gray for random baseline
    }

    # Font sizes
    FONTSIZE_TITLE = 14
    FONTSIZE_LABEL = 12
    FONTSIZE_TICK = 10
    FONTSIZE_LEGEND = 10

    @classmethod
    def apply(cls) -> None:
        """
        Apply SmallML style to matplotlib globally.

        This sets default parameters for all subsequent plots.
        Matches the style used in previous steps.
        """
        # Set default figure size and DPI
        plt.rcParams['figure.figsize'] = cls.FIGSIZE_MAIN
        plt.rcParams['figure.dpi'] = 100  # Screen DPI
        plt.rcParams['savefig.dpi'] = cls.DPI  # Save DPI

        # Font settings
        plt.rcParams['font.size'] = cls.FONTSIZE_TICK
        plt.rcParams['axes.titlesize'] = cls.FONTSIZE_TITLE
        plt.rcParams['axes.labelsize'] = cls.FONTSIZE_LABEL
        plt.rcParams['xtick.labelsize'] = cls.FONTSIZE_TICK
        plt.rcParams['ytick.labelsize'] = cls.FONTSIZE_TICK
        plt.rcParams['legend.fontsize'] = cls.FONTSIZE_LEGEND

        # Grid style
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linestyle'] = '--'

        # Line widths
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.linewidth'] = 1

        # Legend
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.8

        # Color cycle (Tab10 colormap)
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            'color',
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        )

    @classmethod
    def create_figure(
        cls,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a figure with SmallML style applied.

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size. If None, uses FIGSIZE_MAIN
        **kwargs
            Additional arguments passed to plt.subplots()

        Returns
        -------
        fig : plt.Figure
            Figure object
        ax : plt.Axes
            Axes object

        Examples
        --------
        >>> fig, ax = SmallMLPlotStyle.create_figure()
        >>> ax.plot(x, y)
        >>> SmallMLPlotStyle.save_figure(fig, 'output.png')
        """
        if figsize is None:
            figsize = cls.FIGSIZE_MAIN

        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        return fig, ax

    @classmethod
    def create_subplots(
        cls,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create subplots with SmallML style applied.

        Parameters
        ----------
        nrows : int, optional (default=1)
            Number of rows
        ncols : int, optional (default=1)
            Number of columns
        figsize : Tuple[float, float], optional
            Figure size. If None, uses FIGSIZE_MAIN
        **kwargs
            Additional arguments passed to plt.subplots()

        Returns
        -------
        fig : plt.Figure
            Figure object
        axes : plt.Axes or np.ndarray of plt.Axes
            Axes object(s)
        """
        if figsize is None:
            figsize = cls.FIGSIZE_MAIN

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        return fig, axes

    @classmethod
    def save_figure(
        cls,
        fig: plt.Figure,
        filepath: str,
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight',
        **kwargs
    ) -> None:
        """
        Save figure with publication quality settings.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        filepath : str
            Output path
        dpi : int, optional
            Resolution. If None, uses cls.DPI (300)
        bbox_inches : str, optional (default='tight')
            Bounding box mode
        **kwargs
            Additional arguments passed to fig.savefig()

        Examples
        --------
        >>> fig, ax = SmallMLPlotStyle.create_figure()
        >>> ax.plot(x, y)
        >>> SmallMLPlotStyle.save_figure(fig, 'results/figures/plot.png')
        """
        if dpi is None:
            dpi = cls.DPI

        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"[OK] Figure saved: {filepath}")

    @classmethod
    def format_axis(
        cls,
        ax: plt.Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        grid: bool = True,
        legend: bool = False
    ) -> None:
        """
        Apply consistent formatting to an axis.

        Parameters
        ----------
        ax : plt.Axes
            Axes to format
        title : str, optional
            Title for the plot
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        grid : bool, optional (default=True)
            Whether to show grid
        legend : bool, optional (default=False)
            Whether to show legend

        Examples
        --------
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y, label='Data')
        >>> SmallMLPlotStyle.format_axis(
        ...     ax, title='My Plot', xlabel='X', ylabel='Y', legend=True
        ... )
        """
        if title:
            ax.set_title(title, fontsize=cls.FONTSIZE_TITLE, fontweight='bold')

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=cls.FONTSIZE_LABEL)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=cls.FONTSIZE_LABEL)

        if grid:
            ax.grid(True, alpha=0.3, linestyle='--')

        if legend:
            ax.legend(fontsize=cls.FONTSIZE_LEGEND, framealpha=0.8)

        # Tick parameters
        ax.tick_params(labelsize=cls.FONTSIZE_TICK)

    @classmethod
    def get_color_cycle(cls, n: int) -> list:
        """
        Get n colors from the standard cycle.

        Parameters
        ----------
        n : int
            Number of colors needed

        Returns
        -------
        colors : list
            List of hex color codes

        Examples
        --------
        >>> colors = SmallMLPlotStyle.get_color_cycle(3)
        >>> for i, color in enumerate(colors):
        ...     ax.plot(x, y[i], color=color)
        """
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [cycle[i % len(cycle)] for i in range(n)]
        return colors


# Apply style on import
SmallMLPlotStyle.apply()
