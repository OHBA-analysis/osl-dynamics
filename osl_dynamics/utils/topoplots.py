"""Classes and functions to make topoplots.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from osl_dynamics.files.scanner import layouts


def available_layouts():
    layout_names = [file.stem for file in sorted(layouts.glob("*.lay"))]
    return layout_names


def get_layout(layout_name):
    layout_names = available_layouts()

    if layout_name not in layout_names:
        raise FileNotFoundError(f"{layout_name} not found.")

    layout = layouts / f"{layout_name}.lay"
    return str(layout)


def available_outlines():
    outline_names = [file.stem for file in sorted(layouts.glob("*.outline"))]
    return outline_names


def get_outline(outline_name):
    outline_names = available_outlines()

    if outline_name not in outline_names:
        raise FileNotFoundError(f"{outline_name} not found.")

    outline = layouts / f"{outline_name}.outline"
    return str(outline)


class Topology:
    """Topology class.

    Parameters
    ----------
    layout : str
        Path to layout file.
    """

    def __init__(self, layout):
        self.layout = None
        self.outline = None

        layout_filename = get_layout(layout)
        self.read_lay(layout_filename)

        try:
            outline_filename = get_outline(layout)
            self.read_outline(outline_filename)
        except FileNotFoundError:
            pass

    @property
    def width(self):
        """Dynamically calculate the bounding width of the topology."""
        return self.max_x - self.min_x

    @property
    def height(self):
        """Dynamically calculate the bounding height of the topology."""
        return self.max_y - self.min_y

    @property
    def sensor_positions(self):
        return self.layout.loc[self.layout["present"], ["x", "y"]].to_numpy()

    @property
    def sensor_deltas(self):
        return self.layout.loc[self.layout["present"], ["width", "height"]].to_numpy()

    @property
    def sensor_names(self):
        return self.layout.loc[self.layout["present"], "channel_name"].to_numpy()

    @property
    def min_x(self):
        return self.layout.loc[self.layout["present"], "x"].min()

    @property
    def min_y(self):
        return self.layout.loc[self.layout["present"], "y"].min()

    @property
    def max_x(self):
        return self.layout.loc[self.layout["present"], "x"].max()

    @property
    def max_y(self):
        return self.layout.loc[self.layout["present"], "y"].max()

    @property
    def channel_ids(self):
        return self.layout.loc[self.layout["present"], "channel_id"].to_numpy()

    def read_lay(self, filename):
        """Read .lay topology files

        Every line in a .lay file represents a sensor. The data is delimited
        by whitespace. The columns are: channel ID, X, Y, width, height, name.

        Parameters
        ----------
        filename: str
            The location of a .lout file

        """

        layout = pd.read_csv(filename, header=None, sep=r"\s+")
        layout.columns = ["channel_id", "x", "y", "width", "height", "channel_name"]
        layout["present"] = True

        self.layout = layout

    def read_outline(self, filename):
        with open(filename) as f:
            self.outline = [
                np.array(
                    [line.split("\t") for line in outline.splitlines() if line != ""]
                ).astype(np.float)
                for outline in f.read().split("-" * 10)
            ]

    def keep_channels(self, channel_names):
        """Remove channels which aren't present in channel_names

        Remove any channels which don't correspond to on in the names provided.
        This is probably a strong case for using Pandas for data storage.

        Parameters
        ----------
        channel_names: list of str
            A list of channel names which are present in the data.
            All others are removed.
        """
        self.layout.loc[~self.layout["channel_name"].isin(channel_names), "present"] = (
            False
        )

    def plot_data(
        self,
        data,
        plot_boxes=False,
        show_names=False,
        title=None,
        show_deleted_sensors=False,
        colorbar=True,
        axis=None,
        cmap="plasma",
        n_contours=10,
    ):
        """Interpolate the data in sensor-space and plot it.

        Given a data vector which corresponds to each sensor in the topology,
        resample the data by interpolating over a grid. Use this data to
        create a contour plot. Also display the sensor locations and head shape.

        Parameters
        ----------
        data : numpy.array or list
            A vector with data corresponding to each sensor.
        plot_boxes : bool, optional
            Plot boxes to display the height and width of sensors,
            rather than just the centers.
        show_names : bool, optional
            Display channel names.
        title : str, optional
            Title for plot.
        show_deleted_sensors : bool, optional
            Plot the sensors which have been deleted, in red.
        colorbar : bool, optional
            Display colorbar
        axis : matplotlib.pyplot.Axes, optional
            matplotlib axis to plot on.
        cmap : str, optional
            Colourmap to use in plot. Defaults to matplotlib's plasma.
        n_contours : int, optional
            Number of contours to use in plot.

        Returns
        -------
        fig : matplotlib.figure
            Figure.
        """
        if axis is None:
            fig, axis = plt.subplots(figsize=(15, 15))
        else:
            fig = axis.get_figure()
        axis.set_aspect("equal")

        # Create a grid over the bounding area of the Topology.
        grid_x, grid_y = np.mgrid[
            self.min_x : self.max_x : 500j, self.min_y : self.max_y : 500j
        ]

        # Interpolate the data over the new grid.
        grid_z = griddata(
            points=self.sensor_positions,
            values=data,
            xi=(grid_x, grid_y),
            method="cubic",
        )

        # Create a filled contour plot over the interpolated data.
        contour_plot = axis.contourf(
            grid_x, grid_y, grid_z, cmap=cmap, alpha=0.7, levels=n_contours
        )

        if self.outline:
            for o in self.outline:
                axis.plot(*o.T, c="k")

        axis.set_ylim(self.min_y - 0.1, self.max_y + 0.1)
        axis.set_xlim(self.min_x - 0.1, self.max_x + 0.1)

        # Draw boxes to represent sensors.
        if plot_boxes:
            for xy, (dx, dy) in zip(self.sensor_positions, self.sensor_deltas):
                rect = plt.Rectangle(
                    xy - 0.5 * np.array([dx, dy]),
                    dx,
                    dy,
                    facecolor="none",
                    edgecolor="black",
                )
                axis.add_patch(rect)
            if show_deleted_sensors:
                deleted_positions = self.layout.loc[
                    ~self.layout["present"], ["x", "y"]
                ].to_numpy()
                deleted_deltas = self.layout.loc[
                    ~self.layout["present"], ["width", "height"]
                ].to_numpy()
                for xy, (dx, dy) in zip(deleted_positions, deleted_deltas):
                    rect = plt.Rectangle(
                        xy - 0.5 * np.array([dx, dy]),
                        dx,
                        dy,
                        facecolor="none",
                        edgecolor="red",
                    )
                    axis.add_patch(rect)

        if show_names:
            for xy, name in zip(self.sensor_positions, self.sensor_names):
                axis.annotate(name, xy)
            if show_deleted_sensors:
                deleted_positions = self.layout.loc[
                    ~self.layout["present"], ["x", "y"]
                ].to_numpy()
                deleted_names = self.layout.loc[~self.layout["present"], "channel_name"]
                for xy, name in zip(deleted_positions, deleted_names):
                    axis.annotate(name, xy, color="red")

        # Hide axis spines.
        plt.setp(plt.gca().spines.values(), visible=False)

        axis.set_xticks([])
        axis.set_yticks([])

        if colorbar:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(contour_plot, cax=cax)
            cax.set_title("fT")

        if title:
            axis.set_title(title, fontsize=20)

        # Plot sensor positions.
        axis.scatter(
            self.sensor_positions[:, 0],
            self.sensor_positions[:, 1],
            c="k",
            s=5,
        )

        if show_deleted_sensors:
            deleted_positions = self.layout.loc[
                ~self.layout["present"], ["x", "y"]
            ].to_numpy()
            axis.scatter(*deleted_positions.T, color="tab:red", zorder=100)

        fig.patch.set_facecolor("white")

        return fig
