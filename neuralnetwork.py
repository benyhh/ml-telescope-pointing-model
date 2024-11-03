import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from plotter import Plotter
import sys
from IPython import embed
from scipy.stats import ks_2samp, randint, uniform
import matplotlib.pyplot as plt
import dataset


random_seed = 412069413
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


def create_folders(run_number):
    
    PATH_SORTPRED   = f'./FinalResultsOptical/Run{run_number}/SortedPrediction/'
    PATH_LOSSCURVE  = f'./FinalResultsOptical/Run{run_number}/LossCurve/'
    PATH_MODEL      = f'./FinalResultsOptical/Run{run_number}/Model/'
    PATH_SAGE       = f'./FinalResultsOptical/Run{run_number}/Sage/'
    PATH_RESULTS    = f'./FinalResultsOptical/Run{run_number}/' 

    for _path in [PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    return PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE, PATH_RESULTS

patches = {
    0: 	None,
    1:	(-75,75,75,15),
    # 2:	(-28,28,76,60),
    3:	(-120,-98,63,16),
    # 4:	(-75,0,75,15),
    # 5: 	(-28,28,57,47),
    # 5: (-0.083, 0.057),
    # 6: (0.195, 0.266),
    7:  (0,90,90,0),
    8:  (90,180,90,0),
    9:  (-180,-90,90,0),
    10: (-90,0,90,90,0),

    } #


feature_lists = {
    'Corr':   ['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TEMP27','TILT1X','WINDDIRECTION',
               'Az_sun','El_sun','SunAboveHorizon','SunAngleDiff','SunAngleDiff_15','SunElDiff',
               'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
               'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1','ROTATIONX_sumdabs2','ACTUALAZ_sumdabs2',
               'TILT1X_sumdabs2','ACTUALEL_sumdabs5','POSITIONX_sumdabs5','ROTATIONX_sumdabs5',
               'DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT', 'DAZ_TOTAL', 'DEL_TOTAL', 'date'],

    'Corr_reduced':['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TEMP27','TILT1X','WINDDIRECTION',
                    'Az_sun','El_sun','SunAboveHorizon','SunAngleDiff','SunAngleDiff_15','SunElDiff',
                    'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
                    'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1',
                    'DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT','date'],

    'Corr_reduced2':['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TILT1X','WINDDIRECTION',
                    'SunAngleDiff','SunAngleDiff_15','SunElDiff','Hour',
                    'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
                    'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1','DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT','date'],
    
    'Corr_reduced3':['ACTUALAZ','ACTUALEL','TEMP1','TILT1X','WINDDIRECTION','SunAngleDiff_15','Hour',
                    'TURBULENCE','WINDDIR DIFF','DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT', 'date'],

    'hp_el1':          ['DEL_TILT', 'WINDDIRECTION', 'POSITIONZ', 'ACTUALAZ', 'HUMIDITY', 'SunAngleDiff_15', 'TILT1X_sumdabs1', 'WINDDIR DIFF', 'TURBULENCE', 'Hour', 'date'],      

    'hp_az0':          ['SunElDiff', 'DEL_TILT', 'DAZ_TILT', 'DAZ_TILTTEMP', 'TILT1X_sumdabs1', 'ACTUALAZ', 'WINDDIRECTION', 'DAZ_DISP', 'SunElDiff', 'ACTUALEL',
                        'TILT1X', 'TURBULENCE', 'ROTATIONX_sumdabs1', 'TEMP1', 'WINDDIR DIFF','date']
    }



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


def MSDLoss(preds, targets):
    squared_distance = torch.sub(preds, targets).pow(2).sum(dim=1)
    mean_squared_distance = squared_distance.mean()
    return mean_squared_distance

def MSE_loss1(y_true, y_pred):
    y_true = y_true.unsqueeze(1)
    return F.mse_loss(y_true, y_pred)

def MSE_sphere(y_true, y_pred):
    mse = torch.mean((y_pred - y_true)**2, dim=1)

    distance_to_center = torch.norm(y_pred, dim=1)
    distance_to_surface = torch.abs(distance_to_center - 1)
    distance_squared = distance_to_surface**2

    loss = torch.mean(mse + 10*distance_squared)
    return loss

def MSE_scaled_loss(y_true, y_pred, y_scaler):

    y_true_unscaled = y_scaler.inverse_transform(y_true.clone().detach().numpy())
    cosine_tensor = torch.cos(y_true[:,1]).unsqueeze(1)
    cosine_tensor = torch.cos(torch.from_numpy(y_true_unscaled[:,1])).unsqueeze(1)

    scaling = torch.ones(y_true.shape[0], 1)
    scaling = torch.cat((cosine_tensor, scaling), dim=1)

    #Calculate the RMS for the two vectors
    x = torch.sub(y_true, y_pred)

    x = torch.mul(x, scaling)
    x = torch.pow(x, 2)
    x = torch.sum(x, 1)
    x = torch.mean(x)

    return x


class CombinedNetworks(nn.Module):
    def __init__(self, linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function, linear_layer_activation = None):
        super(CombinedNetworks, self).__init__()
        activation_dictionary = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'celu': nn.CELU(),
        }
        

        self.activation_function = activation_dictionary[activation_function]        
        self.linear_layer_activation = linear_layer_activation

        layers = [nn.Linear(nonlinear_input_dim, hidden_layers[0]), self.activation_function]
        layers[0].weight.data.normal_(0, np.sqrt(2.0 / nonlinear_input_dim))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(self.activation_function)
            layers[-2].weight.data.normal_(0, np.sqrt(2.0 / hidden_layers[i]))

        self.nonlinear_layers = nn.Sequential(*layers)
        

        if linear_layer_activation is not None:
            linear_layer = [nn.Linear(linear_input_dim,linear_input_dim)] 
            linear_layer[0].weight.data.normal_(0, np.sqrt(2.0 / (linear_input_dim)))
            linear_layer.append(activation_dictionary[linear_layer_activation])
            self.linear_layer = nn.Sequential(*linear_layer)


        self.final_layer = nn.Linear(linear_input_dim + hidden_layers[-1], 2)
        self.final_layer.weight.data.normal_(0, np.sqrt(2.0 / (linear_input_dim + hidden_layers[-1])))

    def forward(self, X):

        linear_input, nonlinear_input = X
        # linear_output = self.linear_layer(linear_input)
        nonlinear_output = self.nonlinear_layers(nonlinear_input)
        
        if self.linear_layer_activation is not None:
            linear_input    = self.linear_layer(linear_input)

        concatenated_output = torch.cat((linear_input, nonlinear_output), dim=1)
        final_output = self.final_layer(concatenated_output)
        
        return final_output

    def freeze_nonlinear_layers(self):
        for param in self.nonlinear_layers.parameters():
            param.requires_grad = False
    
    def unfreeze_nonlinear_layers(self):
        for param in self.nonlinear_layers.parameters():
            param.requires_grad = True
        
    def freeze_linear_layers(self):
        for param in self.linear_layer.parameters():
            param.requires_grad = False
            
    def unfreeze_linear_layers(self):
        for param in self.linear_layer.parameters():
            param.requires_grad = True



class CombinedNetworksConnected(nn.Module):
    def __init__(self, linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function):
        super(CombinedNetworksConnected, self).__init__()
        activation_dictionary = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'celu': nn.CELU(),
        }
        

        self.activation_function = activation_dictionary[activation_function]        
        
        layers = [nn.Linear(nonlinear_input_dim, hidden_layers[0]), self.activation_function]
        layers[0].weight.data.normal_(0, np.sqrt(2.0 / nonlinear_input_dim))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(self.activation_function)
            layers[-2].weight.data.normal_(0, np.sqrt(2.0 / hidden_layers[i]))
            
        self.nonlinear_layers = nn.Sequential(*layers)
        
        final_layers = [nn.Linear(linear_input_dim + hidden_layers[-1], linear_input_dim + hidden_layers[-1]), nn.Linear(linear_input_dim + hidden_layers[-1], 2)]
        final_layers[0].weight.data.normal_(0, np.sqrt(2.0 / (linear_input_dim + hidden_layers[-1])))
        final_layers[1].weight.data.normal_(0, np.sqrt(2.0 / (linear_input_dim + hidden_layers[-1])))

        self.final_layer = nn.Sequential(*final_layers)


    def forward(self, X):

        linear_input, nonlinear_input = X
        
        nonlinear_output = self.nonlinear_layers(nonlinear_input)
        
        concatenated_output = torch.cat((linear_input, nonlinear_output), dim=1)
        
        final_output = self.final_layer(concatenated_output)
        
        return final_output
    
    def freeze_nonlinear_layers(self):
        for param in self.nonlinear_layers.parameters():
            param.requires_grad = False
    
    def unfreeze_nonlinear_layers(self):
        for param in self.nonlinear_layers.parameters():
            param.requires_grad = True
        
    def freeze_linear_layers(self):
        for param in self.linear_layer.parameters():
            param.requires_grad = False
            
    def unfreeze_linear_layers(self):
        for param in self.linear_layer.parameters():
            param.requires_grad = True



class PINN(nn.Module):
    #write class for pytorch newural network
    def __init__(self, input_size, output_size, hidden_layers, dropout=0.5, activation='relu',scaler = None):
        super().__init__()

        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=dropout)

        self.scaler = scaler

        activation_dictionary = {
            'relu': F.relu,
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU('tanh'),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'celu': nn.CELU(),
        }

        self.activation = activation_dictionary[activation]

    def forward(self, x):
        # # Forward pass through the network, returns the output logits

        for each in self.hidden_layers:
            x = self.activation(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return x




class NeuralNetwork(nn.Module):
    #write class for pytorch newural network
    def __init__(self, input_size, output_size, hidden_layers, dropout=0, activation_function='relu'):
        super().__init__()
      

        activation_dictionary = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'celu': nn.CELU(),
        }

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.activation_function = activation_dictionary[activation_function]


        layers = [nn.Linear(input_size, hidden_layers[0]), self.activation_function]
        layers[0].weight.data.normal_(0, np.sqrt(2. / input_size))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(self.activation_function)
            layers[-2].weight.data.normal_(0, np.sqrt(2.0 / hidden_layers[i]))

        layers.append(nn.Linear(hidden_layers[-1], output_size))


        self.hidden_layers = nn.Sequential(*layers)

        

    def forward(self, x):
        # Forward pass through the network, returns the output logits

        x = self.hidden_layers(x)
    
        return x



class PrepareData():

    def __init__(self,
                df_path              = './Data/merged_features3_all.csv',
                target               = 'both',
                selected_columns_key = 'Corr_reduced2', 
                patch_key            = 0,
                n_components         = 0.99
                ) -> None:

        targets = {
            "total": ["Offset"],
            "az"   : ["Off_Az"],
            "el"   : ["Off_El"],
            "both" : ["Off_El", "Off_Az"]
            }


        self.n_targets = len(targets[target])
        self.df = pd.read_csv(df_path)

        patch            = patches[patch_key]
        selected_columns = feature_lists[selected_columns_key]

        polluted      = False
        self.polluted = polluted

        if patch is not None:
            self.filter_patch(patch)

        if selected_columns is not None:
            self.df = self.df.loc[ : , self.df.columns.isin(selected_columns)]

        self.selected_columns_key = selected_columns_key
        self.patch_key            = patch_key

        self.target         = target
        self.targets        = targets
        self.scaled         = False


        df_pointing = pd.read_csv('./Data/PointingTable.csv') # df with offsets
        df_pointing.insert(0, 'Offset', np.sqrt(df_pointing['Off_El']**2 + df_pointing['Off_Az']**2))

        self.instruments = list(df_pointing['rx'].unique())
        self.df = self.df.merge(df_pointing.loc[: , ['obs_date', 'ca', 'ie', 'rx'] + targets[target]], how = 'left', left_on='date', right_on='obs_date')
        dummies = pd.get_dummies(self.df['rx'])
        self.df = pd.concat([self.df.loc[: , self.df.columns != 'rx'], dummies], axis = 1)



        self.df = self.df.loc[ : , self.df.columns != 'obs_date']
        self.df = self.df.drop_duplicates(subset = ['date'], keep = 'first')
        
        #Removes actualaz and actualel, and replaces them with cartesian coordinate
        self.use_cartesian()

        if polluted:
            self.df.insert(0, 'polluted_az', self.df['Off_Az'])
            self.df.insert(0, 'polluted_el', self.df['Off_El'])

        self.remove_outliers()
        # self.remove_outliers(from_target=False)
        self.train_test_split_days()

        X_train, y_train = self.split_df(self.df_train, target = targets[target])
        X_test, y_test   = self.split_df(self.df_test , target = targets[target])

        X_train, X_test, y_train, y_test = self.scale_data(X_train, X_test, y_train, y_test)
        X_train, X_test                  = self.PCA(X_train, X_test, n_components)

        #Turn x and y into attributes
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test


    def PCA(self, X_train, X_test, n_components, inverse_transform = False):
        #sklearn implementation of PCA
        if inverse_transform is True:
            X_train = self.pca.inverse_transform(X_train)
            X_test  = self.pca.inverse_transform(X_test)
            
        else:
            print('Before PCA:',X_train.shape)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X_train)
            X_train = self.pca.transform(X_train)
            X_test  = self.pca.transform(X_test)
            print('After PCA:',X_train.shape)
    
        return X_train, X_test

    def use_cartesian(self):
        df = self.df
        df['ACTUALAZ'] = np.deg2rad(df['ACTUALAZ'])
        df['ACTUALEL'] = np.deg2rad(df['ACTUALEL'])

        x = np.sin(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
        y = np.cos(df['ACTUALAZ'].values) * np.cos(df['ACTUALEL'].values)
        z = np.sin(df['ACTUALEL'].values)

        #Insert x,y and z, and remove ACTUALAZ and ACTUALEL from df
        df.insert(0, 'X', x)
        df.insert(0, 'Y', y)
        df.insert(0, 'Z', z)

        df = df.loc[ : , ~df.columns.isin(['ACTUALAZ', 'ACTUALEL'])]

        self.df = df


    def filter_patch(self, patch: tuple, rotation = 23):
        """
        Filters self.df to only include data from a patch
        - If len(patch) is 4 -> Filters from left right top bottom with az and el
        - If len(patch) is 2 -> Transform into cartesian coordinates, rotate around 
          x-axis such that the lines are perpendicular to y-axis, then filter between the two y-values.
        
        """
        df = self.df

        if len(patch) == 4:
            l,r,t,b = patch
            df.insert(0, 'ACTUALAZ CUT', df['ACTUALAZ'])
            df.loc[df['ACTUALAZ CUT'] >  180, 'ACTUALAZ CUT'] -= 360
            df.loc[df['ACTUALAZ CUT'] < -180, 'ACTUALAZ CUT'] += 360
            df = df.loc[ (df['ACTUALAZ CUT'] > l) & (df['ACTUALAZ CUT'] < r) & (df['ACTUALEL'] > b) & (df['ACTUALEL'] < t) ]
            df = df.loc[ : , df.columns != 'ACTUALAZ CUT']

        elif len(patch) == 2:
            Az = np.deg2rad(df['ACTUALAZ'])
            El = np.deg2rad(df['ACTUALEL'])

            #x = np.sin(Az.values) * np.cos(El.values)
            y = np.cos(Az.values) * np.cos(El.values)
            z = np.sin(El.values)  

            #x = x
            y = y * np.cos(np.deg2rad(rotation)) - z * np.sin(np.deg2rad(rotation))
            #z = y * np.sin(np.deg2rad(rotation)) + z * np.cos(np.deg2rad(rotation))

            df.insert(0, 'y', y)
            df = df.loc[ (df['y'] > patch[0]) & df['y'] < patch[1], df.columns != 'y' ]

        self.df = df
        return

    def train_test_split_days(self):

        df = self.df
        df['date']  = pd.to_datetime(df['date'])
        df.insert(0, 'day', df['date'].dt.date)
        dfs = [df[df['day'] == day] for day in df['day'].unique()]
        random.Random(random_seed).shuffle(dfs)

        train_size = 0.75
        n_days     = len(dfs)

        dfs_train = dfs[:int(train_size * n_days)]
        dfs_test  = dfs[int(train_size * n_days):]

        self.df_train = pd.concat(dfs_train)
        self.df_test  = pd.concat(dfs_test)

        self.df_train = self.df_train.loc[: , self.df_train.columns != 'day']
        self.df_test  = self.df_test.loc [: , self.df_test.columns  != 'day']
        train_days = len(self.df_train)
        test_days  = len(self.df_test)
        print(f'Training days: {train_days} | Test days: {test_days} | Train size: {train_days/(train_days+test_days):.2f}')
        return

    def remove_outliers(self, from_target = True):
        non_val_cols = ['Hour', 'date']
        if from_target:
            factor = 2
            non_val_cols = ['Off_El', 'Off_Az'] # one or more

        else:
            factor = 1.7
            non_val_cols = list(self.df.loc[: , ~self.df.columns.isin(['Off_Az', 'Off_El', 'Hour', 'date', 'SunAboveHorizon','ie','ca','TEMP1','TEMP27','TILT1T'] + self.instruments)].columns)

        Q1 = self.df.loc[: , self.df.columns.isin(non_val_cols)].quantile(0.25)
        Q3 = self.df.loc[: , self.df.columns.isin(non_val_cols)].quantile(0.75)
        IQR = Q3 - Q1

        ## Will raise ValueError in the future
        self.df = self.df[~((self.df.loc[: , self.df.columns.isin(non_val_cols)] < (Q1 - factor * IQR)) |(self.df.loc[: , self.df.columns.isin(non_val_cols)] > (Q3 + factor * IQR))).any(axis=1)]
        
        return

    def split_df(self, df, target):
        X = df.loc[:, ~ df.columns.isin( ['date'] + target )]
        self.xcols = X.columns
        y = df.loc[:, target]
        return X, y


    def scale_data(self, X_train, X_test, y_train, y_test, scaler = 'PowerTransformer'):
        print(f"Scaling data with {scaler}")
        
        scaler_dict = {'StandardScaler': StandardScaler(), 'PowerTransformer': PowerTransformer()}

        self.scaler1 = scaler_dict[scaler]
        self.scaler2 = scaler_dict[scaler]

        X_train = self.scaler1.fit_transform(X_train.values)
        X_test = self.scaler1.transform(X_test.values)

        if self.n_targets > 1:
            y_train = self.scaler2.fit_transform(y_train.values)
            y_test = self.scaler2.transform(y_test.values)

        else:
            y_train = self.scaler2.fit_transform(y_train.values.reshape(-1,1)).ravel()
            y_test = self.scaler2.transform(y_test.values.reshape(-1,1)).ravel()

        self.scaled = True

        return X_train, X_test, y_train, y_test


    def rescale_data(self, y):
        print('Rescaling data for evaluation')

        if self.n_targets > 1:
            y = self.scaler2.inverse_transform(y)
            
        else:
            y = self.scaler2.inverse_transform(y.reshape(-1,1)).ravel()

        self.scaled = False
        
        return y


class MyDataset(Dataset):

    def __init__(self, X, y):

        #Turn x and y to numpy arrays if they are no
        if not isinstance(X, np.ndarray):
            X = X.values
        if not isinstance(y, np.ndarray):
            y = y.values

        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyCombinedDataset(Dataset):
    def __init__(self, X_linear, X_nonlinear, y):
        
        if not isinstance(X_linear, np.ndarray):
            X_linear = X_linear.values
        if not isinstance(X_nonlinear, np.ndarray):
            X_nonlinear = X_nonlinear.values
        if not isinstance(y, np.ndarray):
            y = y.values
        
        if not torch.is_tensor(X_linear):
            self.X_linear = torch.from_numpy(X_linear)

        if not torch.is_tensor(X_nonlinear):
            self.X_nonlinear = torch.from_numpy(X_nonlinear)
        
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        
    def __len__(self):
        return self.X_linear.size(0)
    
    def __getitem__(self, idx):
        return (self.X_linear[idx], self.X_nonlinear[idx]), self.y[idx]


def train(model, train_loader, test_loader, params, PATH_MODEL='', PATH_LOSSCURVE=''):
    
    FULL_PATH_MODEL = os.path.join(PATH_MODEL, f'model_{params["name"]}.pt') 

    num_epochs = params["num_epochs"]
    learning_rate = params['learning_rate']
    best_val_measure = 1e6
    loss_func = params['loss_func']

    plotter = Plotter(name=params['name'], path = PATH_LOSSCURVE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    all_losses = []
    for e in range(num_epochs):
        batch_losses = []

        #Train both part of network for 50, freeze nonlinear part for 50, then freeze linear part for 50, then ony finetune the last layer
        hasMultipleInputs = isinstance(train_loader.dataset, MyCombinedDataset)
        if e == 50 and hasMultipleInputs:
            model.freeze_nonlinear_layers()
        elif e == 100 and hasMultipleInputs:
            model.freeze_linear_layers()
            model.unfreeze_nonlinear_layers()
        elif e == 150 and hasMultipleInputs:
            model.freeze_nonlinear_layers()
            
        
        for ix, (Xb, yb) in enumerate(train_loader):
            if isinstance(Xb, list):
                _X = (Xb[0].float(), Xb[1].float())
            else:
                _X = Xb.float()

            _y = yb.float()


            #==========Forward pass===============

            preds = model(_X)
            loss = loss_func(_y, preds)     
            #==========backward pass==============

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.data)
            all_losses.append(loss.data)


        mbl = np.mean(batch_losses)

        if e % 5 == 0:
            model.eval()
            test_batch_losses = []
            for _X, _y in test_loader:

                if isinstance(_X, list):
                    _X = (_X[0].float(), _X[1].float())
                else:
                    _X = _X.float()
                
                _y = _y.float()

                #apply model
                test_preds = model(_X)
                test_loss = loss_func(_y, test_preds)
                test_batch_losses.append(test_loss.data)

            mvl = np.mean(test_batch_losses)
            print(f"Epoch [{e+1}/{num_epochs}], Batch loss: {mbl}, Val loss: {mvl}")
            if mvl < best_val_measure:
                best_val_measure = mvl
                torch.save(model.state_dict(), FULL_PATH_MODEL)

            plotter.update_withval(e, mbl, mvl, 'val')
            model.train()

        else:
            print(f"Epoch [{e+1}/{num_epochs}], Batch loss: {mbl}")
            plotter.update(e, mbl, 'train')

    model.load_state_dict(torch.load(FULL_PATH_MODEL))
    model.eval()
    return model



def plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED):
    print(f"Plotting sorted predictions for {params['name']}")

    FULL_PATH_SORTPRED = os.path.join(PATH_SORTPRED, f'sortpred_{params["name"]}.png')
    FULL_PATH_MODEL    = os.path.join(PATH_MODEL, f'model_{params["name"]}.pt')

    

    y_true, y_pred = [], []

    
    for ix, (Xb, yb) in enumerate(test_loader):
        if isinstance(Xb, list):
            _X = (Xb[0].float(), Xb[1].float())
        else:
            _X = Xb.float()

        _y = yb.float()


        if isinstance(Xb, list):
            _X = (Xb[0].float(), Xb[1].float())
        else:
            _X = Xb.float()

        _y = yb.float()
        
        #apply model
        test_preds = model(_X)

        y_true.append(_y.data.numpy())
        y_pred.append(test_preds.data.numpy())

    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if hasattr(ds, "y_scaler"):
        y_true = ds.rescale_y(y_true)
        y_pred = ds.rescale_y(y_pred)

    y_true = np.rad2deg(y_true) * 3600
    y_pred = np.rad2deg(y_pred) * 3600


    plt.clf() 
    n_targets = y_true.shape[1]
    if n_targets > 1:
        x_res = y_true[:,0]-y_pred[:,0]    
        y_res = y_true[:,1]-y_pred[:,1]
    
        RMS_az = np.sqrt(np.mean(x_res**2))
        RMS_el = np.sqrt(np.mean(y_res**2))
        mean_loss = np.sqrt(np.mean(np.linalg.norm(np.stack([x_res, y_res], axis = 1), axis = 1)**2))
        
        no_prediction = np.mean(np.linalg.norm(y_true, axis = 1))
        idxSorted = [y_true[:,i].argsort() for i in range(n_targets)]

        for i, _target in zip(range(n_targets), ['Azimuth', 'Elevation']):
            plt.plot(range(len(y_pred[:,i])), y_pred[idxSorted[i],i], label=f"Predicted {_target}")
            
            plt.plot(range(len(y_pred[:,i])), y_true[idxSorted[i],i], label=f'True {_target}')

    elif n_targets == 1:
        mean_loss     = np.mean(np.abs(y_true-y_pred))
        no_prediction = np.mean(np.abs(y_true))
        
        idxSorted = y_true.argsort()

        plt.plot(range(len(y_pred)), y_pred[idxSorted], label="Prediction")
        
        plt.plot(range(len(y_pred)), y_true[idxSorted], label='True')

    else:
        print('Number of targets not valid')

    plt.xlabel("Sample #")
    plt.ylabel("Offset [arcseconds]")
    print(f'Azimuth RMS: {RMS_az:.3f} arcsecs | Elevation RMS: {RMS_el:.3f}')
    print(f'RMS: {mean_loss:.3f} arcsecs')
    plt.title(f"Neural Network | RMS: {mean_loss:.2f} arcsecs")
    plt.legend()
    
    plt.savefig(FULL_PATH_SORTPRED, dpi = 400)

    return RMS_az, RMS_el, mean_loss



def pred_single(model, x):
    return model(x)


def parameter_sampling():
    """
    Returns randomly sampled parameters for model.
    """

    parameter_space ={
        'hidden_layers': randint.rvs(20, 120, size=randint.rvs(1, 3)),
        'activation': ['gelu', 'tanh', 'relu'],
        'learning_rate': uniform.rvs(0.001, 0.02),
        'batch_size': randint.rvs(32, 512),
        'loss_func': [nn.MSELoss(), MSDLoss],
    }

    params = {}

    for key in parameter_space.keys():
        if type(parameter_space[key]) is list:
            params[key] = random.choice(parameter_space[key])
        else:
            params[key] = parameter_space[key]

    return params


def statistics_for_raw_dataset():

    path = 'SimilarityStatistics/'

    if not os.path.exists(path):
        os.makedirs(path)

    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    relevant_features = nonlinear_features + harmonic_features + geometrical_features + constant_features
    
    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)

    num_bins = 20

    for j in range(n_folds):
        df_test = df.iloc[split_indices[j] : split_indices[j+1]]
        df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]])
        df_trainval = df_trainval[relevant_features]
        
        df_train, df_val = train_test_split(df_trainval.copy(), test_size=0.25, random_state=42)
        
        df_test = df_test[relevant_features]
        df_trainval = df_trainval[relevant_features]

        ds = dataset.PrepareDataCombined(
                df = df_trainval.copy(),
                nonlinear_features = constant_features + nonlinear_features,
                linear_features = geometrical_features + harmonic_features
            )

        print(f'Fold {j}')
        # compare distributions using KS test
        results = []
        for col in df_test.columns:

            min_val = df[col].min()
            max_val = df[col].max()
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

            statistic, pvalue = ks_2samp(df_test[col], df_trainval[col])
            results.append({'feature': col, 'statistic': statistic, 'pvalue': pvalue})

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            ax.hist(df_train[col], density=True, bins = bin_edges, alpha=0.5, label='train')
            ax.hist(df_val[col], density=True, bins = bin_edges, alpha=0.5, label='val')
            ax.hist(df_test[col], density=True, bins = bin_edges, alpha=0.5, label='test')

            ax.legend()
            ax.set_title(f'{col} distribution comparison')

            plt.savefig(f'{path}{col}_fold{j}_distribution_comparison.png', dpi=300)

        # sort results by p-value
        results_sorted = sorted(results, key=lambda x: x['pvalue'])

        # print results
        for result in results_sorted:
            print(f"{result['feature']} - KS statistic: {result['statistic']:.4f}, p-value: {result['pvalue']:.4f}")



def NN_experiment_CV(run_number = 99):

    PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE, PATH_RESULTS = create_folders(run_number)


    feature_list = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1',
                        'HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5',
                        'CA', 'NPAE']

    constant_features = ['COMMANDAZ', 'COMMANDEL']

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)


    for i in range(100):

        num_features = np.random.randint(1, len(feature_list))
        #randomly select num_features from nonlinear_features
        feature_sample = np.random.choice(feature_list, num_features, replace=False).tolist()

        params = parameter_sampling()

        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        params['num_epochs'] = 200
        params['Model #'] = i
        params['hidden_layers'] = [list(hidden_layers)]
        params['features'] = [feature_sample]

        for j in range(n_folds-1, n_folds):

            df_test = df.iloc[split_indices[j] : split_indices[j+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 

            ds = dataset.PrepareDataNN(
                df = df_trainval.copy(),
                features = constant_features + feature_sample,
                run_number = run_number,
            )


            params['name'] = f'regular_rn{run_number}_i{i}_fold{j}'
            

            train_set = MyDataset(ds.X_train, ds.y_train)
            test_set  = MyDataset(ds.X_test, ds.y_test)

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)
            
            input_size    = ds.X_train.shape[1]
            output_size   = ds.y_train.shape[1]

            model = NeuralNetwork(input_size = input_size, output_size = output_size, hidden_layers = hidden_layers, activation_function = activation_function)

            model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
            
            rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

            params['fold'] = j
            params['RMS Val Az'] = rms_val_az
            params['RMS Val El'] = rms_val_el
            params['RMS Val'] = rms_val_tot

            #Convert df to  torch tensor
            X_test = df_test[constant_features + feature_sample].values
            y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

            #Scale with ds.X_scaler and ds.y_scaler
            X_test = ds.X_scaler.transform(X_test)
        
            #Convert to torch tensor
            X_test = torch.from_numpy(X_test).float()

            y_pred = model(X_test)
            y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


            #Convert from radians to arcseconds, and calculate RMS
            y_pred = y_pred * 3600 * 180 / np.pi
            y_test = y_test * 3600 * 180 / np.pi

            rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
            rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

            #Calculate the rms of the magnitude of y_pred - y_test
            rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
            
            params['RMS Test Az'] = rms_test_az
            params['RMS Test El'] = rms_test_el
            params['RMS Test'] = rms_test_tot


            if i == 0:
                df_results = pd.DataFrame(params, index=[0])
            else:
                df_results = df_results.append(params, ignore_index=True)

            df_results.to_csv(f'{PATH_RESULTS}regular_rn{run_number}.csv', index=False)

def combined_separate_experiment_CV(run_number = 99):

    PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE, PATH_RESULTS = create_folders(run_number)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    



    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)


    for i in range(100):

        num_features = np.random.randint(1, 19)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()

        params = parameter_sampling()

        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        params['num_epochs'] = 200
        params['Model #'] = i
        params['hidden_layers'] = [list(hidden_layers)]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]
        
        for j in range(n_folds-1, n_folds):

            df_test = df.iloc[split_indices[j] : split_indices[j+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 

            ds = dataset.PrepareDataCombined(
                df = df_trainval.copy(),
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )

            params['name'] = f'comb_sep_nolayer_rn{run_number}_i{i}_fold{j}'

            train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
            test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

            linear_input_dim    = ds.X_linear_train.shape[1]
            nonlinear_input_dim = ds.X_nonlinear_train.shape[1]

            model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)
            model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
            
            rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

            params['fold'] = j
            params['RMS Val Az'] = rms_val_az
            params['RMS Val El'] = rms_val_el
            params['RMS Val'] = rms_val_tot

            #Convert df to  torch tensor
            ds_test = dataset.PrepareDataCombined( 
                df = df_test,
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )

            X_linear_test = df_test[geometrical_features + harmonic_features].values
            X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
            y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

            X_linear_test = ds.lin_scaler.transform(X_linear_test)
            X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

            X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
            y_pred = model(X_test)
            y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


            #Convert from radians to arcseconds, and calculate RMS
            y_pred = y_pred * 3600 * 180 / np.pi
            y_test = y_test * 3600 * 180 / np.pi

            rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
            rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

            #Calculate the rms of the magnitude of y_pred - y_test
            rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
            
            params['RMS Test Az'] = rms_test_az
            params['RMS Test El'] = rms_test_el
            params['RMS Test'] = rms_test_tot


            if i == 0:
                df_results = pd.DataFrame(params, index=[0])
            else:
                df_results = df_results.append(params, ignore_index=True)

            df_results.to_csv(f'{PATH_RESULTS}comb_sep_nolayer_rn{run_number}.csv', index=False)


def combined_separate_nonlinear_experiment_CV(run_number = 99):

    PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE, PATH_RESULTS = create_folders(run_number)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    

    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    n = len(df)
    n_folds = 6
    split_indices = get_split_indices(n, n_folds)


    for i in range(100):

        num_features = np.random.randint(1, 19)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()

        params = parameter_sampling()

        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        params['num_epochs'] = 200
        params['Model #'] = i
        params['hidden_layers'] = [list(hidden_layers)]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]
     

        for j in range(n_folds):


            df_test = df.iloc[split_indices[j] : split_indices[j+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 

            #print min and max index of test and tainval


            ds = dataset.PrepareDataCombined(
                df = df_trainval.copy(),
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )


            params['name'] = f'comb_sep_nonlin_rn{run_number}_i{i}_fold{j}'

            train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
            test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

            linear_input_dim    = ds.X_linear_train.shape[1]
            nonlinear_input_dim = ds.X_nonlinear_train.shape[1]

            model = CombinedNetworks(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function, linear_layer_activation=activation_function)
            model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
            
            rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

            params['fold'] = j
            params['RMS Val Az'] = rms_val_az
            params['RMS Val El'] = rms_val_el
            params['RMS Val'] = rms_val_tot

            #Convert df to  torch tensor
            ds_test = dataset.PrepareDataCombined( 
                df = df_test,
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )

            X_linear_test = df_test[geometrical_features + harmonic_features].values
            X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
            y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values

            X_linear_test = ds.lin_scaler.transform(X_linear_test)
            X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

            X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
            y_pred = model(X_test)
            y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())


            #Convert from radians to arcseconds, and calculate RMS
            y_pred = y_pred * 3600 * 180 / np.pi
            y_test = y_test * 3600 * 180 / np.pi

            rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
            rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

            #Calculate the rms of the magnitude of y_pred - y_test
            rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
            
            params['RMS Test Az'] = rms_test_az
            params['RMS Test El'] = rms_test_el
            params['RMS Test'] = rms_test_tot


            if i == 0:
                df_results = pd.DataFrame(params, index=[0])
            else:
                df_results = df_results.append(params, ignore_index=True)

            df_results.to_csv(f'{PATH_RESULTS}comb_sep_nonlin_rn{run_number}.csv', index=False)



def combined_connected_experiment_CV(run_number = 99):

    PATH_SORTPRED, PATH_LOSSCURVE, PATH_MODEL, PATH_SAGE, PATH_RESULTS = create_folders(run_number)

    df = pd.read_csv('./Data/dataset_optical_v2.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['C'] = 1
    


    nonlinear_features = ['DAZ_TILT_MEDIAN_1', 'TILT1T_MEDIAN_1', 'DAZ_DISP_MEDIAN_1', 'WINDSPEED_VAR_5', 'TILT1Y_MEDIAN_1',
                        'WINDDIRECTION_MEDIAN_1', 'DEL_TILTTEMP_MEDIAN_1', 'DAZ_TILTTEMP_MEDIAN_1',
                        'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'POSITIONY_MEDIAN_1',
                        'TEMPERATURE_MEDIAN_1', 'POSITIONZ_MEDIAN_1', 'TEMP6_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                        'DAZ_TOTAL_MEDIAN_1', 'DEWPOINT_MEDIAN_1', 'PRESSURE_MEDIAN_1','ROTATIONX_MEDIAN_1']

    harmonic_features = ['HECE', 'HECE2','HECE3','HECE4','HECE5', 'HESE', 'HESE2','HESE3','HESE4','HESE5']
    geometrical_features = ['CA', 'NPAE','C']
    constant_features = ['COMMANDAZ', 'COMMANDEL']

    n = len(df)
    n_folds = 6 
    split_indices = get_split_indices(n, n_folds)


    for i in range(100):

        num_features = np.random.randint(1, 19)
        #randomly select num_features from nonlinear_features
        nonlinear_sample = np.random.choice(nonlinear_features, num_features, replace=False).tolist()

        params = parameter_sampling()

        hidden_layers       = params['hidden_layers']
        activation_function = params['activation']

        params['num_epochs'] = 200
        params['Model #'] = i
        params['hidden_layers'] = [list(hidden_layers)]
        params['nonlinear_features'] = [nonlinear_sample]
        params['linear_features'] = [geometrical_features + harmonic_features]
        
        for j in range(n_folds):


            df_test = df.iloc[split_indices[j] : split_indices[j+1]]
            df_trainval = pd.concat([df.iloc[:split_indices[j]], df.iloc[split_indices[j+1]:]]) 
            
            ds = dataset.PrepareDataCombined(
                df = df_trainval.copy(),
                nonlinear_features = constant_features + nonlinear_sample,
                linear_features = geometrical_features + harmonic_features
            )


            params['name'] = f'comb_conn_rn{run_number}_i{i}_fold{j}'
            

            train_set = MyCombinedDataset(ds.X_linear_train, ds.X_nonlinear_train, ds.y_train)
            test_set  = MyCombinedDataset(ds.X_linear_test,  ds.X_nonlinear_test,  ds.y_test)

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            test_loader  = DataLoader(test_set , batch_size=params['batch_size'], shuffle=True)

            linear_input_dim    = ds.X_linear_train.shape[1]
            nonlinear_input_dim = ds.X_nonlinear_train.shape[1]

            model = CombinedNetworksConnected(linear_input_dim, nonlinear_input_dim, hidden_layers, activation_function)
            model = train(model, train_loader, test_loader, params, PATH_MODEL, PATH_LOSSCURVE)
            
            rms_val_az, rms_val_el, rms_val_tot = plot_sorted_predictions_final(model, test_loader, ds, params, PATH_MODEL, PATH_SORTPRED)

            params['fold'] = j
            params['RMS Val Az'] = rms_val_az
            params['RMS Val El'] = rms_val_el
            params['RMS Val'] = rms_val_tot

            X_linear_test = df_test[geometrical_features + harmonic_features].values
            X_nonlinear_test = df_test[constant_features + nonlinear_sample].values
            y_test = df_test[['OFFSETAZ', 'OFFSETEL']].values
            #Scale with ds.X_scaler and ds.y_scaler
            X_linear_test = ds.lin_scaler.transform(X_linear_test)
            X_nonlinear_test = ds.nonlin_scaler.transform(X_nonlinear_test)

            X_test = (torch.from_numpy(X_linear_test).float(), torch.from_numpy(X_nonlinear_test).float())
            y_pred = model(X_test)
            y_pred = ds.y_scaler.inverse_transform(y_pred.detach().numpy())

            #Convert from radians to arcseconds, and calculate RMS
            y_pred = y_pred * 3600 * 180 / np.pi
            y_test = y_test * 3600 * 180 / np.pi

            rms_test_az = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0])**2))
            rms_test_el = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1])**2))

            #Calculate the rms of the magnitude of y_pred - y_test
            rms_test_tot = np.sqrt( np.mean( np.linalg.norm(y_pred - y_test, axis=1)**2 ) )
            
            params['RMS Test Az'] = rms_test_az
            params['RMS Test El'] = rms_test_el
            params['RMS Test'] = rms_test_tot


            if i == 0:
                df_results = pd.DataFrame(params, index=[0])
            else:
                df_results = df_results.append(params, ignore_index=True)

            df_results.to_csv(f'{PATH_RESULTS}comb_conn_rn{run_number}.csv', index=False)










if __name__ == "__main__":

    if int(sys.argv[1]) == 1:
        NN_experiment_CV(run_number = 98)
    
    if int(sys.argv[1]) == 2:
        combined_separate_experiment_CV(run_number = 98)
    
    if int(sys.argv[1]) == 3:
        combined_separate_nonlinear_experiment_CV(run_number = 98)
    
    if int(sys.argv[1]) == 4:
        combined_connected_experiment_CV(run_number = 98)

    