from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import importlib
import settings
importlib.reload(settings)
from IPython import embed
random_seed = 412069413


class PrepareDataNN():
    def __init__(self,
                df,
                features = None,
                run_number = None,
                ) -> None:


        # embed(header = 'combined ds')
        X = df[features]
        y = df[['OFFSETAZ', 'OFFSETEL']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.scale_data()

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_data(self):
        
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # embed(header = 'scale data')
        self.X_train = self.X_scaler.fit_transform(self.X_train)
        self.X_test = self.X_scaler.transform(self.X_test)
        self.y_train = self.y_scaler.fit_transform(self.y_train)
        self.y_test = self.y_scaler.transform(self.y_test)

    def rescale_y(self, y):
        return self.y_scaler.inverse_transform(y)


class PrepareDataCombined():
    def __init__(self,
                df,
                nonlinear_features = None,
                linear_features = None,
                run_number = None,
                scale_data = True
                ) -> None:


        # embed(header = 'combined ds')
        X_linear = df[linear_features]
        X_nonlinear = df[nonlinear_features]

        y = df[['OFFSETAZ', 'OFFSETEL']]

        X_linear_train, X_linear_test, y_train, y_test = train_test_split(X_linear, y, test_size=0.25, random_state=42)

        X_nonlinear_train = df.loc[X_linear_train.index][nonlinear_features]
        X_nonlinear_test = df.loc[X_linear_test.index][nonlinear_features]

        self.X_linear_train = X_linear_train
        self.X_linear_test = X_linear_test
        self.X_nonlinear_train = X_nonlinear_train
        self.X_nonlinear_test = X_nonlinear_test
        self.y_train = y_train
        self.y_test = y_test

        if scale_data:
            self.scale_data()
        else:
            self.X_linear_train = X_linear_train.values
            self.X_linear_test = X_linear_test.values
            self.X_nonlinear_train = X_nonlinear_train.values
            self.X_nonlinear_test = X_nonlinear_test.values
            self.y_train = y_train.values
            self.y_test = y_test.values

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_data(self):
        
        self.lin_scaler = StandardScaler()
        self.nonlin_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.X_linear_train = self.lin_scaler.fit_transform(self.X_linear_train)
        self.X_linear_test = self.lin_scaler.transform(self.X_linear_test)
        self.X_nonlinear_train = self.nonlin_scaler.fit_transform(self.X_nonlinear_train)
        self.X_nonlinear_test = self.nonlin_scaler.transform(self.X_nonlinear_test)
        self.y_train = self.y_scaler.fit_transform(self.y_train)
        self.y_test = self.y_scaler.transform(self.y_test)

    def rescale_y(self, y):
        return self.y_scaler.inverse_transform(y)



