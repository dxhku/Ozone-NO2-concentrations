import os
import sqlite3
import time
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from numba import njit, prange
from scipy.optimize import fmin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def count_inter_from_start(years, doys):
    """
   Calculate the number of days from a reference start date (START_DAY)
   for a given array of years and day of year (doy) values.

   Parameters:
   ----------
   years : array-like
   doys : array-like

   Returns:
   -------
   numpy.ndarray
       An array of the number of days from the reference start date (START_DAY)
       to each (year, doy) pair in the input.
   """
    days_list = []
    for i in range(0, years.shape[0]):
        year, doy = int(years[i]), int(doys[i])
        jan1 = datetime.date(year, 1, 1)
        date = jan1 + datetime.timedelta(days=doy-1)
        days_list.append((date - START_DAY).days + 1)

    return np.array(days_list)


def expand_date_range_to_df(date_range):
    """
    Expands a list containing the start and end date (year and day of year)
    into a DataFrame of all the days in that range.

    Parameters:
    ----------
    date_range : list of lists
        A list where each sublist contains [year, day_of_year],
        representing the start and end date.

    Returns:
    -------
    df : DataFrame
        A DataFrame containing 'year' and 'day' columns.
    """
    start_year, start_doy = date_range[0]
    end_year, end_doy = date_range[1]

    # Convert year and day of year to datetime objects
    start_date = datetime.datetime.strptime(f'{start_year}-{start_doy}', '%Y-%j')
    end_date = datetime.datetime.strptime(f'{end_year}-{end_doy}', '%Y-%j')

    # Create a list to hold all dates
    all_dates = []

    # Loop through each day from start to end date
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        day_of_year = current_date.timetuple().tm_yday
        all_dates.append([year, day_of_year])
        current_date += timedelta(days=1)

    # Convert to DataFrame
    df = pd.DataFrame(all_dates, columns=['year', 'day'])

    return df


def normalize_data(input_data, params_data):
    """
   Normalize input data based on specified normalization parameters for each feature.

   Parameters:
   ----------
   input_data : pandas.DataFrame
   params_data : list
       A list of parameter objects need to be normalized.

   Returns:
   -------
   pandas.DataFrame
       The input dataframe with normalized values for specified columns.
   """
    for param in params_data:
        original_min, original_max = param.normal_params[0]
        original_abs = max(abs(original_min), abs(original_max))
        if original_min * original_max < 0:
            original_min, original_max = -original_abs, original_abs
        target_min, target_max = param.normal_params[1]

        # Standardization formula
        input_data[param.name] = ((input_data[param.name] - original_min) / (original_max - original_min)) * (
                    target_max - target_min) + target_min
    return input_data


def denormalize_data(input_data, params_data):
    for param in params_data:
        original_min, original_max = param.normal_params[0]
        original_abs = max(abs(original_min), abs(original_max))
        if original_min * original_max < 0:
            original_min, original_max = -original_abs, original_abs
        target_min, target_max = param.normal_params[1]

        # Reverse standardization formula
        input_data[param.name] = ((input_data[param.name] - target_min) / (target_max - target_min)) * (
                original_max - original_min) + original_min

    return input_data


def random_cross_validation_split(data_df, n_splits=10, random_state=42):
    """
    Randomly shuffle and split the dataset into train and test sets for cross-validation.

    Parameters:
    ----------
    data_df : pandas.DataFrame
        The input dataset as a pandas DataFrame, containing features and target columns.
    n_splits : int, optional
        The number of splits (folds) for cross-validation, default is 10.
    random_state : int, optional
        Random seed to ensure reproducibility, default is 42.

    Returns:
    -------
    generator
        A generator yielding a tuple of `(train_data, test_data)` for each fold.
    """
    np.random.seed(random_state)
    data_shuffled = data_df.sample(frac=1).reset_index(drop=True)

    split_size = len(data_shuffled) // n_splits
    for i in range(n_splits):
        start_index = i * split_size
        if i == n_splits - 1:
            end_index = len(data_shuffled)
        else:
            end_index = start_index + split_size

        test_data = data_shuffled.iloc[start_index:end_index]
        train_data = pd.concat([data_shuffled.iloc[:start_index], data_shuffled.iloc[end_index:]])

        test_data = test_data.copy()
        test_data['cv_group'] = i + 1

        yield train_data, test_data


@njit
def st_distance(a_space, a_time, b_space, b_time, scale):
    """
   Calculates the spatiotemporal distance between two sets of points.

   Parameters:
   ----------
   a_space, a_time : numpy.ndarray
       2D array of spatial coordinates (x, y), time for set A.
   b_space, b_time : numpy.ndarray
       2D array of spatial coordinates (x, y), time for set B.
   scale : float
       Scaling factor for the spatial distances.

   Returns:
   -------
   st_dists : numpy.ndarray
       2D array of spatiotemporal distances.
   """
    space_dists = np.sqrt(
        np.square(a_space[:, np.newaxis, 0] - b_space[np.newaxis, :, 0]) +
        np.square(a_space[:, np.newaxis, 1] - b_space[np.newaxis, :, 1])
    )

    time_dists = np.abs(a_time[:, np.newaxis] - b_time[np.newaxis, :])

    st_dists = scale * space_dists + time_dists

    return st_dists


@njit(parallel=True)
def gtwr_chunk(predict_points, known_points, params, x_num):
    """
    Compute the Geographically and Temporally Weighted Regression (GTWR) predictions for a chunk of points.

    Parameters:
    ----------
    predict_points, known_points : numpy.ndarray
    params : tuple
        A tuple containing the scale parameter for spatial distance and the number of nearest neighbors (q).
    x_num : int
        The number of features used for prediction (excluding the intercept term).

    Returns:
    -------
    numpy.ndarray
        A 1D array containing the predicted values for each point in `predict_points`.
    """
    scale, q = params
    q = int(q)  # Ensure q is an integer
    n = predict_points.shape[0]  # Number of prediction points

    # Compute the spatiotemporal distance matrix between prediction and known points
    distance_matrix = st_distance(predict_points[:, 3:5], predict_points[:, -1],
                                  known_points[:, 3:5], known_points[:, -1], scale)

    predictions = np.empty(n)

    for i in prange(n):  # Parallel loop for each prediction point
        distance = distance_matrix[i, :]
        distance[distance == 0] = np.inf  # Avoid division by zero for identical points

        # Get indices of q nearest neighbors
        smallest_indices = np.argsort(distance)[:q]
        max_ds = distance[smallest_indices[-1]]  # Maximum distance among nearest neighbors
        distance = distance[smallest_indices]

        # Compute weight matrix W based on the distances
        W = np.exp(-np.square(distance) / (max_ds ** 2))
        W_diag = np.diag(W)

        # Construct the design matrix X with intercept (first column = 1)
        X = np.empty((q, x_num + 1))
        X[:, 0] = 1  # Intercept term
        X[:, 1:] = known_points[smallest_indices, 5:x_num + 5]  # Use feature columns

        y = known_points[smallest_indices, -2]  # Target variable (assumed to be in the second-to-last column)

        # Perform weighted least squares regression
        XTWX = X.T @ W_diag @ X
        XTWy = X.T @ W_diag @ y
        beta = np.linalg.pinv(XTWX) @ XTWy  # Compute regression coefficients

        # Predict value for the current point
        X_with_intercept = np.concatenate((np.array([1]), predict_points[i, 5:x_num + 5]))
        y_pred = X_with_intercept @ beta
        predictions[i] = y_pred

    return predictions


@njit
def gtwr(predict_points, known_points, params, x_num, batch_size=1000):
    """
    Perform GTWR over a large set of points, processing in batches.

    Parameters:
    ----------
    predict_points, known_points : numpy.ndarray
    params : tuple
        A tuple containing the scale for spatial distances and the number of nearest neighbors (q).
    x_num : int
        The number of features used for prediction (excluding the intercept term).
    batch_size : int, optional
        The number of points to process in each batch (default is 1000).

    Returns:
    -------
    numpy.ndarray
        A 1D array of predicted values for all points in `predict_points`.
    """
    scale, q = params
    q = int(q)

    # Handle invalid parameters by returning a large error value
    if q <= 10 or scale <= 0:
        return np.full(predict_points[:, 3].shape[0], 1e10)

    n = predict_points.shape[0]
    predictions = np.empty(n)

    # Process points in batches to save memory
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        predict_points_chunk = predict_points[start:end]
        chunk_predictions = gtwr_chunk(predict_points_chunk, known_points, params, x_num)
        predictions[start:end] = chunk_predictions

    return predictions


def gtwr_cv(points, cv, params, x_num, y_name):
    """
    Perform cross-validation or fitting result for GTWR model.

    Parameters:
    ----------
    points : pandas.DataFrame
    cv : int
    params : tuple
    x_num : int
    y_name : str

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the GTWR predictions for each point, with cross-validation group labels if applicable.
    """
    start_time = time.time()
    all_results = pd.DataFrame()

    # Calculate for fitting result
    if cv == 1:
        print(f'\rCalculate fitting result for sample data.', end='', flush=True)
        test_result = gtwr(points.to_numpy().astype(np.float64),
                           points.to_numpy().astype(np.float64),
                           params, x_num)

        # Create a copy of the group and store the GTWR predictions
        all_results = points.copy()
        all_results['gtwr'] = test_result

    # Calculate for verify result
    else:
        count = 0
        # Iterate through training and test splits generated by a random CV splitter
        for train_group, test_group in random_cross_validation_split(points, cv):
            count += 1
            print(f'\r{params}\tfor CV {count}', end='', flush=True)

            # Perform GTWR for the test set
            test_result = gtwr(test_group.to_numpy().astype(np.float64)[:, :-1],
                               train_group.to_numpy().astype(np.float64),
                               params, x_num)

            # Create a copy of the test group and store the GTWR predictions
            result_df = test_group.copy()
            result_df['gtwr'] = test_result
            result_df['cv_group'] = test_group['cv_group'].values

            # Concatenate the results of each fold to the final DataFrame
            all_results = pd.concat([all_results, result_df], ignore_index=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Compute the Root Mean Squared Error (RMSE) between actual and predicted values
    rmse = np.sqrt(mean_squared_error(all_results[y_name], all_results['gtwr']))
    print(f'\t{params}\ttime: {elapsed_time}s, RMSE: {rmse:.4f}')

    return all_results


def save_accurancy_pic(data1, data2, save_path, pics, models=None):
    real_name, pred_name, param_name = pics
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if models:
        fig.subplots_adjust(bottom=0.2)

    titles = ["Verification Results", "Fitting Results"]
    data_list = [data1, data2]

    for i, (ax, data, title) in enumerate(zip(axes, data_list, titles)):
        actual = data[real_name].values
        predicted = data[pred_name].values

        reg = LinearRegression()
        reg.fit(actual.reshape(-1, 1), predicted)

        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        sns.scatterplot(x=actual, y=predicted, color='blue', s=10, alpha=0.5, edgecolor=None, ax=ax)
        ax.plot([0, max(predicted)], [0, max(predicted)], 'k-', label='1:1 line')

        x_vals = np.array(ax.get_xlim())
        y_vals = reg.predict(x_vals.reshape(-1, 1))
        ax.plot(x_vals, y_vals, 'c--', label=f"Y = {reg.coef_[0]:.2f}X + {reg.intercept_:.2f}")

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, min(max(actual), max(predicted)))
        ax.set_ylim(0, min(max(actual), max(predicted)))

        ax.text(0.05, 0.95, f'N = {len(actual)}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.90, f'R² = {r2:.2f}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.85, f'RMSE = {rmse:.2f}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.80, f'MAE = {mae:.2f}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.75, f'MAPE = {mape:.2f}%', fontsize=10, transform=ax.transAxes)

        ax.set_xlabel(f'Actual {param_name} (μg/m³)')
        ax.set_ylabel(f'Predicted {param_name} (μg/m³)')
        ax.set_title(title)
        ax.legend()

    if models:
        opt, aux_list = models
        st_scale, q_num = opt
        plt.figtext(0.5, 0.08, f'Optimal params -- scale: {st_scale}, q: {q_num}', ha="center", fontsize=10)
        plt.figtext(0.5, 0.05, 'Auxiliary variables-- %s' % ', '.join(aux_list), ha="center", fontsize=10)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


class Database:
    def __init__(self, db):
        self.db = db

    def create_table(self, table_name, fields):
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        try:
            create_sql = f'''
               CREATE TABLE IF NOT EXISTS {table_name} (
                    {','.join([' '.join(field) for field in fields])}
               )
           '''
            cursor.execute(create_sql)
            conn.commit()
        except Exception as e:
            print(f'Error executing SQL: {e}')
            conn.rollback()
        finally:
            conn.close()

    def update_table(self, table_name, update_items):
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        try:
            for update_item in update_items:
                alter_clause = ', '.join([f"{col} = ?" for col, _ in update_item['alter']])
                condition_clause = ' AND '.join([f"{col} = ?" for col, _ in update_item['condition']])
                update_sql = f"UPDATE {table_name} SET {alter_clause} WHERE {condition_clause}"
                params = [value for _, value in update_item['alter']] + [value for _, value in update_item['condition']]
                cursor.execute(update_sql, params)
            conn.commit()
        except Exception as e:
            print(f'Error executing SQL: {e}')
            conn.rollback()
        finally:
            conn.close()

    def insert_table(self, table_name, insert_items):
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        try:
            for insert_item in insert_items:
                columns = ', '.join([col for col, _ in insert_item])
                placeholders = ', '.join(['?' for _ in insert_item])
                values = [value for _, value in insert_item]
                insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                cursor.execute(insert_sql, values)
            conn.commit()
        except Exception as e:
            print(f'Error executing SQL: {e}')
            conn.rollback()
        finally:
            conn.close()

    def execute_sql(self, sql_sen):
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_sen)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]  # 获取列名
            df = pd.DataFrame(rows, columns=columns)
            conn.commit()
            return df
        except Exception as e:
            print(f'Error executing SQL: {e}')
            conn.rollback()
        finally:
            conn.close()


class AuxVar:
    def __init__(self, name, time_sign, time_gap, normal_params, nan_data=None):
        self.name = name
        self.time_sign = time_sign
        self.time_gap = time_gap
        self.normal_params = normal_params
        if nan_data:
            self.nan_data = nan_data


class GTWR:
    def __init__(self, name, base_dir, predict_item, aux_var_list):
        self.name = name
        self.base_dir = base_dir

        self.db = Database(os.path.join(base_dir, fr"{self.name}.db"))

        self.shp_dir = os.path.join(base_dir, "shp")
        self.shp_grid_point = os.path.join(self.shp_dir, "grid_points.shp")
        self.shp_station_point = os.path.join(self.shp_dir, "station_points.shp")
        self.minx, self.miny, self.maxx, self.maxy = gpd.read_file(self.shp_grid_point).total_bounds

        self.raster_dir = os.path.join(base_dir, "raster")

        self.result_dir = os.path.join(base_dir, "result")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.accuracy_dir = os.path.join(base_dir, "accuracy")
        if not os.path.exists(self.accuracy_dir):
            os.makedirs(self.accuracy_dir)

        self.predict_item = predict_item
        self.aux_var_list = aux_var_list
        self.aux_var_size = len(self.aux_var_list)
        self.aux_var_names = [aux_var.name for aux_var in self.aux_var_list]

        self.st_var_list, self.known_points = None, None

    def prepare_sample_data(self):
        # Create SQL sentence and execute
        x_array, y_array = [aux_var.name for aux_var in self.aux_var_list], [self.predict_item]
        sql = 'select id, year, day, lon, lat, {} from sample_data'.format(','.join(x_array + y_array))
        sql_point = self.db.execute_sql(sql)

        sql_point['times'] = count_inter_from_start(sql_point['year'].values, sql_point['day'].values)

        # Handling null values
        sql_point = sql_point.dropna()
        for param in self.aux_var_list:
            if getattr(param, 'nan_data', None):
                sql_point = sql_point[~sql_point[[param.name]].isin([param.nan_data]).any(axis=1)]

        if NORMALIZE_SIGN is None:
            return sql_point
        else:
            self.st_var_list = [AuxVar('lon', None, None, normal_params=[[self.minx, self.maxx], [0, 1]]),
                                AuxVar('lat', None, None, normal_params=[[self.miny, self.maxy], [0, 1]]),
                                AuxVar('times', None, None,
                                       normal_params=[[sql_point['times'].min(), sql_point['times'].max()], [0, 1]])]
            if NORMALIZE_SIGN == 'st':
                return normalize_data(sql_point, self.st_var_list)
            else:
                return normalize_data(sql_point, self.st_var_list + self.aux_var_list)

    def objective_function(self, params):
        predict = gtwr_cv(self.known_points, 10, params, self.aux_var_size, self.predict_item)
        return mean_squared_error(predict[self.predict_item], predict['gtwr'])

    def extract_data_from_grid(self, group, time):
        grid_points = gpd.read_file(self.shp_grid_point).drop(columns=['geometry'])
        year, day = group['year'].values[0], group['day'].values[0]
        grid_points.insert(1, 'year', year)
        grid_points.insert(2, 'day', day)

        for param in self.aux_var_list:
            name = param.name
            time_sign, time_gap = param.time_sign, param.time_gap
            raster_dir = os.path.join(self.raster_dir, name)
            file_time_sign = f'y{year}' if time_sign == 'year' else f'y{year}_d{day-(day%time_gap)+time_gap//16}'

            raster_file = os.path.join(raster_dir, f"{name}_{file_time_sign}.tif")
            if os.path.exists(raster_file):
                with rasterio.open(raster_file) as raster:
                    lon, lat = grid_points['lon'].values, grid_points['lat'].values
                    rows, cols = raster.index(lon, lat)
                    raster_data = raster.read(1)
                    raster_values = raster_data[rows, cols]
                    grid_points[param.name] = raster_values

        desired_order = ['id', 'year', 'day', 'lon', 'lat'] + [aux_var.name for aux_var in self.aux_var_list]
        grid_points = grid_points[desired_order]
        grid_points['times'] = time

        # Handling null values
        grid_points = grid_points.dropna()
        for param in self.aux_var_list:
            if getattr(param, 'nan_data', None):
                grid_points = grid_points[~grid_points[[param.name]].isin([param.nan_data]).any(axis=1)]

        if NORMALIZE_SIGN is None:
            return grid_points
        elif NORMALIZE_SIGN == 'st':
            return normalize_data(grid_points, self.st_var_list)
        else:
            return normalize_data(grid_points, self.st_var_list + self.aux_var_list)

    def gtwr_grid(self, params, time_range=None):
        gtwr_dir = os.path.join(self.result_dir, 'gtwr')
        if not os.path.exists(gtwr_dir):
            os.makedirs(gtwr_dir)

        sample_points_time = self.db.execute_sql('select distinct year, day from sample_data order by year, day')
        sample_points_time['times'] = count_inter_from_start(sample_points_time['year'].values, sample_points_time['day'].values)

        predict_points_time = expand_date_range_to_df(time_range) if time_range else sample_points_time
        predict_points_time['times'] = count_inter_from_start(predict_points_time['year'].values, predict_points_time['day'].values)

        grouped = predict_points_time.groupby('times')
        for time, group in list(grouped):
            year, day = group['year'].values[0], group['day'].values[0]
            print(f'\rCalculate GTWR grid for {year}/{day} -- Extract Aux from raster', end='', flush=True)
            predict_points = self.extract_data_from_grid(group, time)

            print(f'\rCalculate GTWR grid for {year}/{day} -- Calculate grid value', end='', flush=True)
            predictions = gtwr(predict_points.to_numpy().astype(np.float64),
                               self.known_points.to_numpy().astype(np.float64),
                               params, self.aux_var_size)

            if NORMALIZE_SIGN is not None:
                predict_points = denormalize_data(predict_points, self.st_var_list)

            coords = predict_points[['lon', 'lat']].values

            geometries = [Point(xy) for xy in coords]
            gdf = gpd.GeoDataFrame({'lon': coords[:, 0], 'lat': coords[:, 1], 'gtwr': predictions.flatten()},
                                   geometry=geometries)
            gdf.set_crs(epsg=4326, inplace=True)

            output_file = os.path.join(gtwr_dir, f"gtwr_y{year}_d{day}.shp")
            gdf.to_file(output_file)


if __name__ == '__main__':
    ''' Static parameters '''
    # The year and the day of the year the sample data starts
    START_DAY = datetime.date(2020, 5, 1)
    # Predict time range [year, day of year]
    PREDICT_TIME_RANGE = [[2021, 1], [2021, 2]]
    # NORMALIZE_SIGN determines the method of normalization:
    # None  - No normalization applied.
    # 'st'  - Only normalize spatial and temporal items.
    # 'all'  - Normalize independent variable and spatiotemporal items.
    NORMALIZE_SIGN = 'all'

    ''' Parameters of auxiliary variables '''
    tno2 = AuxVar(name='tno2', time_sign='day', time_gap=1, normal_params=[[0, 2000], [0, 1]])
    temp = AuxVar(name='temp', time_sign='day', time_gap=1, normal_params=[[270, 315], [0, 1]])
    et = AuxVar(name='et', time_sign='day', time_gap=1, normal_params=[[-0.3, 0.3], [-1, 1]])
    sp = AuxVar(name='sp', time_sign='day', time_gap=1, normal_params=[[70000, 11000], [0, 1]])
    tp = AuxVar(name='tp', time_sign='day', time_gap=1, normal_params=[[0, 5], [0, 1]])
    ws = AuxVar(name='ws', time_sign='day', time_gap=1, normal_params=[[0, 30], [0, 1]])
    pop = AuxVar(name='pop', time_sign='year', time_gap=1, normal_params=[[0, 50000], [0, 1]])
    building = AuxVar(name='building', time_sign='year', time_gap=1, normal_params=[[0, 5000000], [0, 1]])
    road = AuxVar(name='road', time_sign='year', time_gap=1, normal_params=[[0, 5000], [0, 1]])
    parking = AuxVar(name='parking', time_sign='year', time_gap=1, normal_params=[[0, 150], [0, 1]])
    ndvi = AuxVar(name='ndvi', time_sign='day', time_gap=16, normal_params=[[-2000, 10000], [-1, 1]], nan_data=-3000)

    ''' Create GTWR object '''
    gtwr_obj = GTWR(name='example',
                    base_dir=r"example_data",
                    predict_item='NO2',
                    aux_var_list=[tno2, temp, et, sp, tp, ws, pop, building, road, parking, ndvi])

    ''' Prepare the sample data'''
    gtwr_obj.known_points = gtwr_obj.prepare_sample_data()

    ''' Calculate the best params for GTWR model '''
    initial_guess = [1, 20]
    optimal_params = fmin(gtwr_obj.objective_function, initial_guess, disp=True)
    print(f"best st_scale: {optimal_params[0]}, best q_num: {int(optimal_params[1])}")
    st_scale_optimal, q_num_optimal = optimal_params[0], int(optimal_params[1])

    ''' Calculate verify matters for GTWR result '''
    verify_result = gtwr_cv(gtwr_obj.known_points, 10, (st_scale_optimal, q_num_optimal),
                            gtwr_obj.aux_var_size, gtwr_obj.predict_item)
    verify_file_path = os.path.join(gtwr_obj.accuracy_dir, 'result_gtwr_verify.csv')
    with open(verify_file_path, 'w', encoding='utf-8') as f:
        f.write(f"scale: {st_scale_optimal}, q: {q_num_optimal}\n")
    verify_result.to_csv(verify_file_path, mode='a', index=False, encoding='utf-8')

    ''' Calculate fitting matters for GTWR result'''
    fitting_result = gtwr_cv(gtwr_obj.known_points, 1, (st_scale_optimal, q_num_optimal),
                             gtwr_obj.aux_var_size, gtwr_obj.predict_item)
    fitting_file_path = os.path.join(gtwr_obj.accuracy_dir, 'result_gtwr_fitting.csv')
    with open(fitting_file_path, 'w', encoding='utf-8') as f:
        f.write(f"scale: {st_scale_optimal}, q: {q_num_optimal}\n")
    fitting_result.to_csv(fitting_file_path, mode='a', index=False, encoding='utf-8')

    ''' Save accuracy picture'''
    accuracy_pic_path = os.path.join(gtwr_obj.accuracy_dir, 'result_gtwr_without.png')
    pic_params = (gtwr_obj.predict_item, 'gtwr', 'NO2')
    model_params = ((st_scale_optimal, q_num_optimal), gtwr_obj.aux_var_names)
    save_accurancy_pic(verify_result, fitting_result, accuracy_pic_path, pic_params, model_params)

    ''' Calculate Grid GTWR'''
    gtwr_obj.gtwr_grid((st_scale_optimal, q_num_optimal), PREDICT_TIME_RANGE)
    print('\n---------- DONE ----------')
