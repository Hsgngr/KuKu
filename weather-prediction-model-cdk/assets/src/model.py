from sklearn.base import clone


class Model:
    def __init__(self, classifier_model, regressor_model, cols):
        self.time_offsets = ["5", "10", "15", "30", "60"]
        self.cols = cols
        self.classifiers = [clone(classifier_model) for _ in range(5)]
        self.regressors = [clone(regressor_model) for _ in range(5)]

    def fit(self, df_timed_dct):
        for i, time_offset in enumerate(self.time_offsets):
            table = df_timed_dct[time_offset]["train"]
            target_cols = ["TARGET_RAIN_PERCENTAGE", "TARGET_WEATHER"]
            table = table[self.cols + target_cols]
            X = table[table.columns[:-2]]
            weather = table[table.columns[-1]]
            rain_percent = table[table.columns[-2]]
            self.classifiers[i].fit(X, weather)
            self.regressors[i].fit(X, rain_percent)

    def predict(self, row):
        if row.isna().sum() > 0:
            return None
        weather_preds = [classifier.predict(
            [row[self.cols]])[0] for classifier in self.classifiers]
        rain_percent_preds = [regressor.predict(
            [row[self.cols]])[0] for regressor in self.regressors]
        return {ts: {'type': int(weather), str(weather): float(rain_percent)}
                for ts, weather, rain_percent in zip(self.time_offsets, weather_preds, rain_percent_preds)}
