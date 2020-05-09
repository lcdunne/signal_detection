# TO DO

- Enable reverse order of signal/noise input (so it doesn't have to be just 'most positive' through 'least positive'. E.g. `ascending` argument as `True` or `False` to specify the order of responses.

- Code to transform classifier data to ROC data. Go from having columns `['y_true', 'y_classified', 'rating']` to signal and noise data for each level of the rating scale.

- Enable ROC analysis from response time data. Can scale with `(RT - RTmin) / (RTmax - RTmin) * 100` for example, then use this as the "decision" scale.
