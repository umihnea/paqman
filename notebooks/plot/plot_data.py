import numpy as np
import pandas


class PlotDataException(RuntimeError):
    pass


class PlotData:
    def __init__(self, data, label, smoothing_window=0):
        self.data = data
        self.label = label
        self.smoothing_window = smoothing_window

    def _smoothen_layer(self, layer):
        return (
            pandas.Series(layer)
            .rolling(self.smoothing_window, min_periods=self.smoothing_window)
            .mean()
            .to_numpy()
        )


class MultiRunData(PlotData):
    def __init__(self, data, label, smoothing_window=0):
        super().__init__(
            self._to_ndarray(data), label, smoothing_window=smoothing_window
        )

    def _to_ndarray(self, data):
        """Cast a multi-level list to a NumPy matrix.
        In case not all lists have the same length, we only get a NumPy
        array of Python lists, which we do not want and will raise.
        """
        for index, layer in enumerate(data):
            if not isinstance(layer, list):
                raise PlotDataException("Layer {} is not a list.".format(index))

        prev_count = len(data[0])
        for layer in data[1:]:
            count = len(layer)
            if count != prev_count:
                raise PlotDataException("NumPy array cannot correctly form.")

        return np.array(data)

    def _make_smoother(self, data):
        if self.smoothing_window == 0:
            return data

        return np.array([self._smoothen_layer(layer) for layer in data])

    def plot(self, ax):
        data = self._make_smoother(self.data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        x_axis = np.array(range(len(data[0])))

        ax.plot(
            x_axis, mean, linestyle="solid", alpha=1, rasterized=True, label=self.label
        )
        ax.fill_between(
            x_axis, mean - std, mean + std, alpha=0.25, linewidth=0.0, rasterized=True
        )


class SingleRunData(PlotData):
    def __init__(self, data, label, smoothing_window=0):
        super().__init__(np.array(data), label, smoothing_window=smoothing_window)

    def _make_smoother(self, data):
        if self.smoothing_window == 0:
            return data

        return self._smoothen_layer(data)

    def plot(self, ax):
        data = self._make_smoother(self.data)
        ax.plot(data, label=self.label)
