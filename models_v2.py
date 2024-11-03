import importlib
import xgboost as xgb
import lightgbm as lgb
from dataset import PrepareDataCombined
from settings import patches, features, dataset_params
from IPython import embed
import pandas as pd
import json
import sage
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error,make_scorer, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression


random_seed = 412069413
MAX_EVALS = 200

# sns.set(font_scale=1.5)

plt.rcParams['axes.titlesize'] = 18; plt.rcParams['axes.labelsize'] = 18;
plt.rcParams["xtick.labelsize"] = 18; plt.rcParams["ytick.labelsize"] = 18; plt.rcParams["legend.fontsize"] = 18


class Model():
    """
    Parent class for all model classes
    Contains common functions like plotting, evaluation,
    and feature importance

    All childclasses need to implement these attributes:
     - dataset
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.params  = dataset.params 
        self.X_train, self.X_test, self.y_train, self.y_test = dataset.get_data()
    

        if self.params is not None:
            if self.params['new_model']:
                self.PATH_MODEL = f'./NewModel/Models/' 
                self.PATH_PLOTS = f'./NewModel/Plots/' 

            elif self.params['optical_model']:
                self.PATH_MODEL = './AnalyticalModelRaw/Models/'
                self.PATH_PLOTS = './AnalyticalModelRaw/Plots/'
              
            
            else:
                self.PATH_MODEL = f'./Results_v3/{dataset.params["patch_key"]}/{dataset.params["feature_key"]}/Models/' 
                self.PATH_PLOTS = f'./Results_v3/{dataset.params["patch_key"]}/{dataset.params["feature_key"]}/Plots/'
                print('Feature key', dataset.params['feature_key'])

        else:
            self.PATH_MODEL = f'./FinalResults/Run{dataset.run_number}/Models/'
            self.PATH_PLOTS = f'./FinalResults/Run{dataset.run_number}/Plots/'

    def get_name(self):
        return NotImplementedError('Must be implented in child model class')

    def plot_sorted_predictions_v2(self):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)


        if self.dataset.target_key == 'testaz':
            pred = self.model.predict(self.X_test)-self.X_test[:,0]
            real = self.y_test - self.X_test[:,0]
        elif self.dataset.target_key == 'testel':
            pred = self.model.predict(self.X_test)-self.X_test[:,1]
            real = self.y_test - self.X_test[:,1]

        pred = np.rad2deg(pred) * 3600
        real = np.rad2deg(real) * 3600
        

        me = mean_absolute_error(real, pred)
        
        embed()

        if self.dataset.target == 'both':
            noPred = np.mean(np.sqrt(self.dataset.df['Off_Az']**2 + self.dataset.df['Off_El']**2))
        else:
            try:
                noPred = np.mean(np.abs(real))
            except:
                embed()
            


        plt.clf()
        plt.figure(figsize=(18,12))


        if self.dataset.n_targets > 1:
            idxSorted0 = self.y_test[:,0].argsort()
            idxSorted1 = self.y_test[:,1].argsort()
        
            plt.plot(range(len(pred)), pred[idxSorted0,0], label="Prediction Az")
            plt.plot(range(len(pred)), pred[idxSorted1,1], label="Prediction El")
            
            
            plt.plot(range(len(pred)), self.y_test[idxSorted0, 0], label='Real Az')
            plt.plot(range(len(pred)), self.y_test[idxSorted1, 1], label='Real El')

        else:
            print('y_test shape: ', self.y_test.shape)
            try:
                idxSorted = real.ravel().argsort()
            except:
                idxSorted = real.argsort()
            plt.plot(range(len(pred)), pred[idxSorted], label="Prediction")
            try:
                plt.plot(range(len(pred)), real[idxSorted], label="Real data")
            except:
                embed()

        plt.xlabel("Sample #")
        plt.ylabel("Offset [arcseconds]")
        
        plt.title(f"{self.name} | ME: {me:.3f} | {noPred:.3f}")
        plt.legend()

        if self.params is not None:
            if self.params['use_pca'] is True:
                save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_pca_{self.params['target']}.png")
                save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_pca_{self.params['target']}.png")
            else:
                save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_{self.params['target']}.png")
                save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_{self.params['target']}.png")
        else:
            save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}.png")
            save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}.png")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()
    def plot_sorted_predictions(self):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)


        
        pred = self.model.predict(self.X_test)
        me = mean_absolute_error(self.y_test, pred)
        
        if self.dataset.target == 'both':
            noPred = np.mean(np.sqrt(self.dataset.df['Off_Az']**2 + self.dataset.df['Off_El']**2))
        else:
            try:
                noPred = self.dataset.df.loc[:, self.dataset.target].abs().values.mean() 
            except:
                embed()
            


        plt.clf()
        plt.figure(figsize=(18,12))


        if self.dataset.n_targets > 1:
            idxSorted0 = self.y_test[:,0].argsort()
            idxSorted1 = self.y_test[:,1].argsort()
        
            plt.plot(range(len(pred)), pred[idxSorted0,0], label="Prediction Az")
            plt.plot(range(len(pred)), pred[idxSorted1,1], label="Prediction El")
            
            
            plt.plot(range(len(pred)), self.y_test[idxSorted0, 0], label='Real Az')
            plt.plot(range(len(pred)), self.y_test[idxSorted1, 1], label='Real El')

        else:
            print('y_test shape: ', self.y_test.shape)
            try:
                idxSorted = self.y_test.ravel().argsort()
            except:
                idxSorted = self.y_test.argsort()
            plt.plot(range(len(pred)), pred[idxSorted], label="Prediction")
            try:
                idxSorted = self.y_test.ravel().argsort()
                plt.plot(range(len(pred)), self.y_test.ravel()[idxSorted], label="Real data")
            except:
                embed()


        me = np.rad2deg(me) * 3600

        plt.xlabel("Sample #")
        plt.ylabel("Offset [arcseconds]")
        
        plt.title(f"{self.name} | ME: {me:.3f} | {noPred:.3f}")
        plt.legend()

        if self.params is not None:
            if self.params['use_pca'] is True:
                save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_pca_{self.params['target']}.png")
                save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_pca_{self.params['target']}.png")
            else:
                save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}_{self.params['target']}.png")
                save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}_{self.params['target']}.png")
        else:
            save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}.png")
            save_path_hist = os.path.join(PATH_HISTOGRAM,  f"histogram_{self.name}.png")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()
    

    def plot_sorted_prediction_v2(self,elevation_for_scaling=None):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)


        
        pred = self.model.predict(self.X_test)
        residual = pred - self.y_test
        if elevation_for_scaling is not None:
            residual *= np.cos(elevation_for_scaling)
        
        RMS = np.sqrt(np.mean( np.rad2deg(residual * 3600)**2 ) )


        plt.clf()
        plt.figure(figsize=(18,12))


        print('y_test shape: ', self.y_test.shape)
        try:
            idxSorted = self.y_test.ravel().argsort()
        except:
            idxSorted = self.y_test.argsort()
        plt.plot(range(len(pred)), pred[idxSorted], label="Prediction")
        try:
            idxSorted = self.y_test.ravel().argsort()
            plt.plot(range(len(pred)), self.y_test.ravel()[idxSorted], label="Real data")
        except:
            embed()


        # me = np.rad2deg(RMS) * 3600
        me = RMS
        print(f'RMS for {self.name}: {me} arcsecs')
        benchmark = self.dataset.benchmark
        nopred = np.mean(np.abs(self.y_test))

        plt.xlabel("Sample #")
        plt.ylabel("Offset [arcseconds]")
        
        plt.title(f"{self.name} | RMS: {me:.3f}\'\' | With corrections: {benchmark:.3f}\'\' | Optical Model: {nopred:.3f}\'\'")
        plt.legend()

        save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}.png")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()

    def plot_sorted_prediction_final(self, X=None, y=None, fn=''):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')

        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)

        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        pred = self.model.predict(X)
        residual = pred - y        
        RMS = np.sqrt( np.mean( residual**2 ) )


        plt.clf()
        plt.figure(figsize=(12,8))


        print('y_test shape: ', y.shape)
        try:
            idxSorted = y.ravel().argsort()
        except:
            idxSorted = y.argsort()
        plt.plot(range(len(pred)), pred[idxSorted], label="Predicted")
        try:
            idxSorted = y.ravel().argsort()
            plt.plot(range(len(pred)), y.ravel()[idxSorted], label="True")
        except:
            embed()

        print(f'RMS for {self.name}: {RMS} arcsecs')
        rms_offset = self.dataset.rms_offset
        rms_offset_optimal_correction = self.dataset.rms_offset_optimal_correction


        plt.xlabel("Sample #")
        plt.ylabel("Offset [\'\']")
        # plt.title(f"{self.name} | RMS: {RMS:.2f}\'\' | With corrections: {rms_offset:.2f}\'\' | Optimal model: {rms_offset_optimal_correction:.2f}\'\'")
        plt.title(f'True and predicted offset')
        plt.legend()

        save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}{fn}.pdf")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()
        return RMS, rms_offset

    def plot_histogram(self, X=None, y=None, fn=''):
        print(f"Plotting histogram for {self.name}")
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_HISTOGRAM):
            os.makedirs(PATH_HISTOGRAM)

        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        pred = self.model.predict(X)

        plt.clf()

        n_bins = 25
        plt.figure(figsize=(12,8))
        #Increase title and lable size
    
        _, bins, _ = plt.hist(y, bins = n_bins, alpha = 0.8, label = 'Current offset')
        plt.hist(y - pred, bins = bins, alpha = 0.8, label='XGB Model')
        plt.xlabel('Offset [\'\']')#, fontsize = fontsize)
        plt.ylabel('Number of samples')#, fontsize = fontsize)
        # plt.xticks(fontsize = fontsize)
        # plt.yticks(fontsize = fontsize)
        plt.title('Distribution of pointing offsets with and without XGB model')
        #plt.legend(fontsize = fontsize)
        plt.legend()
        plt.tight_layout()

        save_path_hist = os.path.join(PATH_HISTOGRAM, f"hist_{self.name}{fn}.pdf")

        plt.savefig(save_path_hist, dpi = 400)


    def plot_error_locations(self):
        print(f"Plotting error locations for {self.name}")
        PATH_ERRORLOC = os.path.join(self.PATH_PLOTS, f'ErrorLocations/')
        if not os.path.exists(PATH_ERRORLOC):
            os.makedirs(PATH_ERRORLOC)


        pred = self.model.predict(self.X_test)
        # Calculate the errors between the true y and predicted y

        errors = np.sqrt((pred - self.y_test)**2)

        # Create a scatter plot of the true y and predicted y
        fig = plt.figure() 
        ax = fig.add_subplot()
        ax.scatter(self.X_train[:,0], self.X_train[:,1], c='blue', label='Training data')

        # Mark the data points with large errors in a different color
        threshold = np.percentile(errors, 90)
        threshold = np.deg2rad(2/3600)
        indices = np.where(errors > threshold)[0]
        print(f'Number of samples with large errors: {len(indices)}, threshold: {threshold:.3f}')

        ax.scatter(self.X_test[indices,0],self.X_test[indices,1], c='red', label='Large error on test data')

        ax.set_xlabel('Azimuth')
        ax.set_ylabel('Elevation')
        plt.legend()

        if self.params is not None:
            if self.params['use_pca'] is True:
                save_path_errloc = os.path.join(PATH_ERRORLOC,  f"errloc_{self.name}_pca_{self.params['target']}.png")
            else:
                save_path_errloc = os.path.join(PATH_ERRORLOC,  f"errloc_{self.name}_{self.params['target']}.png")
        else:
            save_path_errloc = os.path.join(PATH_ERRORLOC,  f"errloc_{self.name}.png")

        plt.savefig(save_path_errloc, dpi = 400)
        plt.clf()
        return


class LinearRegressor(LinearRegression):
    def __init__(self, target = 'LinearRegressor', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target

    def fit(self, X, y):
        return super().fit(X, y)
    
    def predict(self, X):
        return super().predict(X)
    
    def plot(self, X, y, elevation_for_scaling = None, figsize=(8, 4)):
        y_pred = self.predict(X)
        residuals = (y - y_pred)

        if elevation_for_scaling is not None:
            residuals *= np.cos(elevation_for_scaling)
        #print mean residuals in degrees
        print(f'RMS {self.target}: {np.sqrt( np.mean( np.rad2deg(residuals* 3600 )**2 ) ):.2f} arcsecs')
        
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        ax1, ax2 = axes
        
        ax1.scatter(y, y_pred)
        ax1.set(xlabel='True Values', ylabel='Predictions', title='True vs. Predicted Values')
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        
        ax2.hist(residuals, bins=25)
        ax2.set(xlabel='Residuals', ylabel='Frequency', title='Residual Histogram')
        
        plt.tight_layout()
        plt.savefig(f'./FinalResultsOptical/Plots/{self.target}_performance.png',dpi = 400)

    def plot_residuals(self, X, y):
        #Make plot with residuals and X[:,0] on the x-axis
        y_pred = self.predict(X)
        residuals = y - y_pred
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], residuals)
        ax.set(xlabel='X', ylabel='Residuals', title='Residuals vs. X')
        plt.tight_layout()
        plt.savefig(f'./FinalResultsOptical/Plots/{self.target}_residuals.png',dpi = 400)
        return


import torch 
def pytorch_model_wrapper(model, input_data):

    input_tensor = torch.from_numpy(input_data).float()
    output_tensor = model(input_tensor)
    output = output_tensor.detach().numpy()
    return output






def prepare_linear_terms(Az,El):
    terms_az = [
        Az,
        np.sin(Az),
        np.cos(3*Az),
        np.sin(2*Az),
        np.cos(2*Az),
        np.cos(Az) * np.tan(El),
        np.sin(Az) * np.tan(El),
        np.tan(El),
        1/np.cos(El),
        np.cos(2*Az) / np.cos(El),
        np.cos(Az) / np.cos(El),
        np.cos(5*Az) / np.cos(El),
        np.ones(len(Az)),
    ]
    terms_el = [
        El,
        np.sin(El),
        np.cos(El),
        np.cos(2*Az),
        np.sin(2*Az),
        np.cos(3*Az),
        np.sin(3*Az),
        np.sin(4*Az),
        np.sin(5*Az),
        np.sin(Az),
        np.sin(Az)*np.tan(El),
        np.ones(len(Az)),
    ]

    #Set of all terms in terma_az and terms_el
    terms_both = [
        Az,
        El,
        np.sin(Az),
        np.cos(3*Az),
        np.sin(2*Az),
        np.cos(2*Az),
        np.cos(Az) * np.tan(El),
        np.sin(Az) * np.tan(El),
        np.tan(El),
        1/np.cos(El),
        np.ones(len(Az)),
        np.sin(El),
        np.cos(El),
        np.sin(3*Az),
        np.sin(4*Az),
        np.sin(5*Az),
        np.cos(2*Az) / np.cos(El),
        np.cos(Az) / np.cos(El),
        np.cos(5*Az) / np.cos(El)
    ]

    terms_az = np.column_stack(terms_az)
    terms_el = np.column_stack(terms_el)
    terms_both = np.column_stack(terms_both)
    
    return terms_az, terms_el, terms_both


def correlation_plot(df):
    """
    Plot the correlation of residuals with the features using seaborn.
    Residuals are 1d numpy array and features are dataframe.
    """
    state = 'ALL'
    df.loc[df.SUNEL_MEDIAN_1 < 0. , 'SUNEL_MEDIAN_1'] = 0.
    df.loc[df.SUNEL_MEDIAN_1 < 0. , 'SUNAZ_MEDIAN_1'] = 0.
    df['HOUR'] = df['date'].dt.hour
    df['MINUTE'] = df['date'].dt.minute
    # df = df[df.SUNEL_MEDIAN_1 > 0]
    print(f'NUMBER OF DATAPOINTS DURING {state} = {len(df)}')
    print(df.describe())
    corr = df.corr(method = 'pearson')
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(25, 18))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, annot_kws={"size": 14})
    plt.savefig(f'./AnalyticalModelRaw/Plots/Correlation_residuals_{state.lower()}.png', dpi = 400)


def fit_linear_models(tp_key = 0):

    path_df = './Data/raw_nflash230.csv'
    dataset_params['target'] = 'optical_both'
    time_period_tests = {
        0: (pd.Timestamp('2022-05-22 06:00:00'), pd.Timestamp('2022-05-22 23:40:00')),
        1: (pd.Timestamp('2022-05-22'), pd.Timestamp('2022-07-04')),
        2: (pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-09-17 17:30:00')),
        3: (pd.Timestamp('2022-03-01'), pd.Timestamp('2022-05-22')),
        4: (pd.Timestamp('2022-07-05'), pd.Timestamp('2022-08-12')),
    }

    start, end = time_period_tests[tp_key]
    
    print(f'tp_key: {tp_key}, start: {start}, end: {end}')
    df = pd.read_csv(path_df)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start) & (df['date'] <= end)]

    df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']] = np.deg2rad(df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']])

    X_train, X_test, y_train, y_test = train_test_split(df[['COMMANDAZ', 'COMMANDEL']], df[['ACTUALAZ', 'ACTUALEL']], test_size=0.2, random_state=random_seed)
    

    X_az_train, X_el_train, X_both_train = prepare_linear_terms(X_train['COMMANDAZ'], X_train['COMMANDEL'])
    X_az_test, X_el_test, X_both_test = prepare_linear_terms(X_test['COMMANDAZ'], X_test['COMMANDEL'])
    
    y_az_train, y_el_train = y_train['ACTUALAZ'], y_train['ACTUALEL']
    y_az_test, y_el_test = y_test['ACTUALAZ'], y_test['ACTUALEL']

    model_az = LinearRegressor(target=f'az_tp{tp_key}')
    model_az.fit(X_az_train, y_train['ACTUALAZ'].values)
    model_az.plot(X_az_train, y_train['ACTUALAZ'].values, X_train['COMMANDEL'])
    model_az.plot_residuals(X_az_train, y_train['ACTUALAZ'].values)

    model_el = LinearRegressor(target='el_datasplit')
    model_el.fit(X_el_train, y_train['ACTUALEL'].values)
    model_el.plot(X_el_train, y_train['ACTUALEL'].values)
    model_el.plot_residuals(X_el_train, y_train['ACTUALEL'].values)

    # Predict on the testing set
    az_pred_test = model_az.predict(X_az_test)
    el_pred_test = model_el.predict(X_el_test)

    az_residuals_test = az_pred_test - y_test['ACTUALAZ'].values
    el_residuals_test = el_pred_test - y_test['ACTUALEL'].values


    az_residuals_test *= np.cos(X_test['COMMANDEL'].values)
    az_residuals_test = np.rad2deg(az_residuals_test) * 3600
    el_residuals_test = np.rad2deg(el_residuals_test) * 3600

    mean_error_test = np.sqrt(np.mean(az_residuals_test**2 + el_residuals_test**2))
    
    print(f'RMS on testing set: {mean_error_test:.2f} arcsecs')

    # Predict on the training set
    az_pred_train = model_az.predict(X_az_train)
    el_pred_train = model_el.predict(X_el_train)

    az_residuals_train = az_pred_train - y_train['ACTUALAZ'].values
    el_residuals_train = el_pred_train - y_train['ACTUALEL'].values

    az_residuals_train *= np.cos(X_train['COMMANDEL'].values)
    az_residuals_train = np.rad2deg(az_residuals_train) * 3600
    el_residuals_train = np.rad2deg(el_residuals_train) * 3600
    mean_error_train = np.sqrt(np.mean(az_residuals_train**2 + el_residuals_train**2))
    
    print(f'RMS on training set: {mean_error_train:.2f} arcsecs')

def linear_model_dump():
    
    df_features = pd.read_csv('./Data/processed_optical/features_optical.csv')
    df_features['date'] = pd.to_datetime(df_features['date'])
    df_features = df_features[df_features.date < pd.Timestamp('2022-09-17')]
    df_features = df_features[feats]
    
    
    df = pd.read_csv('./Data/raw_nflash230.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df.date < pd.Timestamp('2022-09-17')]

    df = df.merge(df_features, on = 'date', how = 'inner')
    df = df.dropna().drop_duplicates()

    date = df.date
    df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']] = np.deg2rad(df[['COMMANDAZ', 'COMMANDEL', 'ACTUALAZ', 'ACTUALEL']])
    print(len(df))
    terms_az, terms_el, terms_both = prepare_linear_terms(df['COMMANDAZ'], df['COMMANDEL'])
    print('Type dt.hour', type(df.date.dt.hour))



    model_az = LinearRegressor(target = 'az_v2')
    model_az.add_terms(terms_az)
    model_az.fit(df['ACTUALAZ'].values)
    model_az.plot(model_az.X, df['ACTUALAZ'].values, df.COMMANDEL.values)
    model_az.plot_residuals(model_az.X, df['ACTUALAZ'].values)
    model_el = LinearRegressor(target = 'el_v2')
    model_el.add_terms(terms_el)
    model_el.fit(df['ACTUALEL'].values)
    model_el.plot(model_el.X, df['ACTUALEL'].values)
    model_el.plot_residuals(model_el.X, df['ACTUALEL'].values)

    az_pred = model_az.predict(model_az.X)
    el_pred = model_el.predict(model_el.X)
    
    az_residuals = az_pred - df['ACTUALAZ'].values
    el_residuals = el_pred - df['ACTUALEL'].values

    az_residuals *= np.cos(df['COMMANDEL'].values) 

    mean_error = np.mean(np.sqrt(az_residuals**2 + el_residuals**2))
    mean_error = np.rad2deg(mean_error) * 3600
    print(f'Mean error: {mean_error:.2f} arcsecs')

    print("Fitted parameters for azimuth:")
    print(model_az.intercept_*3600)
    print(model_az.coef_*3600)
    print("Fitted parameters for elevation:")
    print(model_el.intercept_*3600)
    print(model_el.coef_*3600)

    df_residuals = pd.DataFrame({'date': date, 'RESIDUALSAZ': az_residuals, 'RESIDUALEL': el_residuals})
    df_residuals['date'] = pd.to_datetime(df_residuals['date'])



    #Merge df_residuals with df_features on date
    df_features = pd.merge(df_features, df_residuals, on = 'date', how = 'inner').dropna().drop_duplicates()
    correlation_plot(df_features)


def plot_sorted_predictions_models(model, X, y_true, y_scaler, name, azimuth = False):
    print(f"Plotting sorted predictions for {name}")

    PATH_SORTPRED = './PretrainedModel/SortedPrediction/'
    FULL_PATH_SORTPRED = os.path.join(PATH_SORTPRED, f'sortpred_{name}.png')
    
    
    y_pred = model.predict(X)

    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true)
        y_pred = y_scaler.inverse_transform(y_pred)

    plt.clf()       

    residuals = y_true - y_pred
    residuals = np.rad2deg(residuals) * 3600
    
    if azimuth:
        residuals *= np.cos(X[:, 1])

    RMS = np.sqrt(np.mean( residuals**2 ))
    
    idxSorted = y_true[:,0].argsort()
    plt.plot(range(len(y_pred)), y_pred[idxSorted], label="Prediction")
    plt.plot(range(len(y_pred)), y_true[idxSorted], label='True')

    plt.xlabel("Sample #")
    plt.ylabel("Offset [arcseconds]")
    print(f'RMS for {name}: {RMS:.3f} arcsecs')
    plt.title(f"{name} | RMS: {RMS:.3f} arcsecs ")
    plt.legend()
    
    plt.savefig(FULL_PATH_SORTPRED, dpi = 400)

    return RMS

def get_split_indices(l, n):
    """
    Returns a list of cumulative indices to split an array of length l into n sub-arrays.
    The sub-arrays contain approximately the same number of rows.
    """
    indices = [0]
    size = l // n
    remainder = l % n
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + size + 1
        else:
            end = start + size
        indices.append(end)
        start = end

    return indices


def linreg_rawdata_cv(run_number = 99):

    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)


    nonlinear_features = []
    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']
    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5']#'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    az_features = ['HASA', 'HSCA2', 'HACA3', 'HASA2', 'HACA2', 'HSCA', 'HSCA5',
                    'C', 'COMMANDAZ', 'COMMANDEL']
    el_features = ['HESE', 'HECE', 'HACA2', 'HASA2', 'HACA3', 'HESA3', 'HESA4',
                    'HESA5', 'C', 'COMMANDEL']

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)

    rms_train_az = np.zeros(n_folds)
    rms_train_el = np.zeros(n_folds)
    rms_train_tot = np.zeros(n_folds)

    rms_test_az = np.zeros(n_folds)
    rms_test_el = np.zeros(n_folds)
    rms_test_tot = np.zeros(n_folds)

    for j in range(n_folds):
        print(f'Fold {j+1}/{n_folds}')
        
        df_test = df.iloc[split_indices[j] : split_indices[j+1]]
        df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 
       
        ds = PrepareDataCombined(
            df = df_trainval,
            nonlinear_features = az_features,
            linear_features = el_features,
            scale_data = False
        )


        #Concatenate features together, ds.X_linear_train and ds.X_nonlinear_train
        X_train_az = ds.X_nonlinear_train
        X_train_el = ds.X_linear_train

        X_val_az = ds.X_nonlinear_test
        X_val_el = ds.X_linear_test
        y_train = np.rad2deg(ds.y_train) * 3600
        y_val = np.rad2deg(ds.y_test) * 3600

        
        X_test_az = df_test[az_features].values
        X_test_el = df_test[el_features].values
        y_test = np.rad2deg(df_test[['OFFSETAZ', 'OFFSETEL']].values) * 3600


        model_az = LinearRegressor(target=f'az_fold{j}')
        model_az.fit(X_train_az, y_train[:,0])

        model_el = LinearRegressor(target=f'el_fold{j}')
        model_el.fit(X_train_el, y_train[:,1])


        y_pred_az = model_az.predict(X_train_az)
        y_pred_el = model_el.predict(X_train_el)

        rms_train_az[j] = np.sqrt( mean_squared_error(y_train[:,0], y_pred_az) )
        rms_train_el[j] = np.sqrt( mean_squared_error(y_train[:,1], y_pred_el) )
        rms_train_tot[j] = np.sqrt( np.mean( np.linalg.norm(y_train - np.stack([y_pred_az, y_pred_el], axis=1), axis=1)**2 ) )
        print(f'AZ RMS: {rms_train_az[j]:.3f}, EL RMS: {rms_train_el[j]:.3f}, Total RMS: {rms_train_tot[j]:.3f}')
        
        y_pred_az = model_az.predict(X_test_az)
        y_pred_el = model_el.predict(X_test_el)

        rms_test_az[j] = np.sqrt( mean_squared_error(y_test[:,0], y_pred_az) )
        rms_test_el[j] = np.sqrt( mean_squared_error(y_test[:,1], y_pred_el) )
        rms_test_tot[j] = np.sqrt( np.mean( np.linalg.norm(y_test - np.stack([y_pred_az, y_pred_el], axis=1), axis=1)**2 ) )

        print(f'AZ RMS: {rms_test_az[j]:.3f}, EL RMS: {rms_test_el[j]:.3f}, Total RMS: {rms_test_tot[j]:.3f}')


    df_results = pd.DataFrame({
        'Fold': np.arange(n_folds),
        'RMS Train Az': rms_train_az,
        'RMS Train El': rms_train_el,
        'RMS Train': rms_train_tot,
        'RMS Test Az': rms_test_az,
        'RMS Test El': rms_test_el,
        'RMS Test': rms_test_tot
    })
    
    df_results.to_csv(f'{PATH_RESULTS}linreg_rn{run_number}.csv', index=False)
    print(df_results.mean())
    embed()


feats = ['date', 'WINDSPEED_VAR_5', 'WINDDIRECTION_MEDIAN_1', 'SUNAZ_MEDIAN_1', 'SUNEL_MEDIAN_1',
    'TILT1T_MEDIAN_1', 'TILT1X_MEDIAN_1', 'TILT1Y_MEDIAN_1',
    'TEMP1_MEDIAN_1', 'TEMP26_MEDIAN_1', 'TEMPERATURE_MEDIAN_1',
    'POSITIONX_MEDIAN_1', 'POSITIONY_MEDIAN_1', 'POSITIONZ_MEDIAN_1',
    'ROTATIONX_MEDIAN_1', 'ROTATIONY_MEDIAN_1', 'HUMIDITY_MEDIAN_1',
    'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1', 'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1',
    'DISP_ABS3_MEDIAN_1']
