import numpy as np
import pytest
from matplotlib.container import BarContainer
from explainlp.plotting import NLPlotter


@pytest.fixture
def plotter():
    topic_names = {0: "foo", 1: "bar", 2: "baz"}
    return NLPlotter(topic_titles=topic_names)


def test_init_plotter(plotter):
    assert plotter.titles == ["foo", "bar", "baz"]


def test_simple_plot(plotter):
    data = np.random.rand(3)
    fig = plotter.plot_doc(data)
    assert isinstance(fig, BarContainer), "Wrong type of plot!"
    assert len(fig.datavalues) == 3, "Wrong number of bars!"


def test_weird_data_plot(plotter):
    """Test a one-dimensional plot"""
    data = np.random.rand(1, 3)
    fig = plotter.plot_doc(data)
    assert isinstance(fig, BarContainer), "Wrong type of plot!"
    assert len(fig.datavalues) == 3, "Wrong number of bars!"


def test_empty_data(plotter):
    """Tests empty data raises error"""
    data = np.random.rand(0)
    with pytest.raises(ValueError) as error_msg:
        fig = plotter.plot_doc(data)
    assert "empty document" in str(error_msg)
