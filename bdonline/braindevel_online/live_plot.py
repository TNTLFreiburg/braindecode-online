from braindevel_online.ring_buffer import RingBuffer
from matplotlib import pyplot as plt
import sys
import argparse
import numpy as np
#import seaborn

# prevent seaborn import
def plot_sensor_signals(signals, sensor_names=None, figsize=None, 
        yticks=None, plotargs=[], sharey=True, highlight_zero_line=True,
        xvals=None,fontsize=9):
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
        "sensor names for all sensor matrices")
    if sensor_names is None:
        sensor_names = map(str, range(len(signals)))  
    num_sensors = signals.shape[0]
    print('DEBUG 2.3.1')
    if figsize is None:
        print('DEBUG 2.3.1.1')
        figsize = (7, np.maximum(num_sensors // 4, 1))
        print('DEBUG 2.3.1.2')
    print('DEBUG 2.3.1.3')
    figure, axes = plt.subplots(num_sensors, sharex=True, sharey=sharey,
        figsize=figsize)
    print('DEBUG 2.3.1.4')
    for sensor_i in range(num_sensors):
        print('DEBUG 2.3.2')
        if num_sensors > 1:
            ax = axes[sensor_i]
        else:
            ax = axes
        if xvals is None:
            ax.plot(signals[sensor_i], *plotargs)
        else:
            ax.plot(xvals, signals[sensor_i], *plotargs)
        if yticks is None:
            ax.set_yticks([])
        elif (isinstance(yticks, list)): 
            ax.set_yticks(yticks)
        elif yticks == "minmax":
            ymin, ymax = ax.get_ylim()
            ax.set_yticks((ymin, ymax - ymax/10.0))
        elif yticks == "onlymax":
            ymin, ymax = ax.get_ylim()
            ax.set_yticks([ymax])
        elif yticks == "keep": 
            pass
        ax.text(-0.035, 0.4, sensor_names[sensor_i], fontsize=fontsize,
            transform=ax.transAxes,
            horizontalalignment='right')
        if (highlight_zero_line):
            # make line at zero
            ax.axhline(y=0,ls=':', color="grey")
        print('DEBUG 2.3.3')
    max_ylim = np.max(np.abs(plt.ylim()))
    plt.ylim(-max_ylim, max_ylim)
    print('DEBUG 2.3.4')
    figure.subplots_adjust(hspace=0)
    print('DEBUG 2.3.5')
    return figure

class LivePlot:
    def __init__(self, plot_freq, range_update_freq=None):
        self._plot_freq = plot_freq
        self._range_update_freq = range_update_freq
        if self._range_update_freq is None:
            self._range_update_freq = self._plot_freq * 6
    
    def showLivePlots(self, intervalInSec, durationInSec):
        self._plotIntervalInSec = intervalInSec
        self._totalDurationInSec = durationInSec
        self._initPlots(self._sensor_names)
        self._continouslyUpdatePlots()
        
    def initPlots(self, sensor_names):
        print('DEBUG 1')
        self._sensor_names = sensor_names
        print('DEBUG 2')
        self._init_figure()
        print('DEBUG 3')
        self._init_data()
        print('DEBUG 4')
        self.init_counters()
        print('DEBUG 5')
        

    def _init_figure(self):
        plt.ion()
        print('DEBUG 2.1')
        plt.show()
        print('DEBUG 2.2')
        n_sensors = len(self._sensor_names)
        print('DEBUG 2.3')
        self.fig = plot_sensor_signals(np.outer(np.ones(n_sensors), 
            np.sin(np.linspace(0,10,1000))),
            self._sensor_names, figsize=(12,12))
        print('DEBUG 2.4')
        self.fig.suptitle("EEG Sensors")
        print('DEBUG 2.5')
        self.fig.canvas.draw()
        print('DEBUG 2.6')

    def _init_data(self):
        self._data = dict()
        for sensor_name in self._sensor_names:
            self._data[sensor_name] = RingBuffer(np.sin(np.linspace(0,10,1000)))
    
    def init_counters(self):
        self.n_samples = 0
        self.i_last_plot = 0
        self.i_last_range_update = 0

    def accept_samples(self, samples):
        """Samples should be in time x chan format"""
        for i_sensor, sensor_name in enumerate(self._sensor_names):
            self._data[sensor_name].extend(samples[:,i_sensor])
        self.n_samples += len(samples)
        if self.n_samples - self.i_last_range_update > self._range_update_freq:
            self._setRangeOfAllPlotsToMaxMin()
            self.i_last_range_update = self.n_samples
        if self.n_samples - self.i_last_plot > self._plot_freq:
            self._updateAllPlots()
            self.i_last_plot = self.n_samples
            
            self.fig.canvas.update()
            self.fig.canvas.flush_events()
        
    def _updateAllPlots(self):
        for sensor_name in self._sensor_names:
            self._updatePlot(sensor_name, self._data[sensor_name])
        #plt.pause(0.001)

    def _updatePlot(self, sensor_name, new_data):
        ax = self.fig.axes[self._sensor_names.index(sensor_name)]
        line = ax.lines[0]
        line.set_ydata(new_data)
        
        ax.draw_artist(line)
    
    def _continouslyUpdatePlots(self):
        self._lastRangeUpdate = 0
        self._setRangeOfAllPlotsToMaxMin()
        i = 0
        while True:
            self._collectPackets()
            if i % 10 == 0:
                self._updateRangeOfPlots()
                self._updateAllPlots()


    def _setRangeOfAllPlotsToMaxMin(self):
        # find max and min deviation from mean
        totalMax = 0
        totalMin = sys.maxint
        sensorMeans = {}
        for sensor_name in self._sensor_names:
            sensorData = self._data[sensor_name]
            sensorMean = int(sum(sensorData) / len(sensorData))
            sensorMinimum = min(sensorData) - sensorMean
            sensorMaximum = max(sensorData) - sensorMean
            sensorMeans[sensor_name] = sensorMean
            if (sensorMinimum < totalMin):
                totalMin = sensorMinimum
            if (sensorMaximum > totalMax):
                totalMax = sensorMaximum
        for sensor_name in self._sensor_names:
            ax = self.fig.axes[self._sensor_names.index(sensor_name)]
            padding = 0.1
            ax.set_ylim(sensorMeans[sensor_name] + totalMin - padding,
              sensorMeans[sensor_name] + totalMax + padding
                )
        

    def close(self):
        plt.close()

def parseCommandLineArguments():
        parser = argparse.ArgumentParser(description='Live Plotting all the sensors :)')
        parser.add_argument('--debug', action="store_true", help="take dummy epoc headset")
        parser.add_argument('--duration', type=int, default=sys.maxint, help='how long to run plotting (in sec)')
        parser.add_argument('--interval', type=int, default=5, help='how much data to show in one plot (in sec)')
        return parser.parse_args()

if __name__ == '__main__':
    args = parseCommandLineArguments()
        
    livePlot = LivePlot()
    livePlot.showLivePlots(args.interval, args.duration)
