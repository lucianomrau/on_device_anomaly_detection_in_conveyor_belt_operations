import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import sed_eval
import dcase_util
import tempfile



if not os.path.exists('my_functions.py'):
    # Clone the specific file from GitHub
    subprocess.run([
        'curl', '-o', 'my_functions.py',
        'https://raw.githubusercontent.com/lucianomrau/TinyML_Anomaly_Detection_for_Industrial_Machines_with_Periodic_Duty_Cycles/master/my_functions.py'
    ])

from my_functions import df_timestamps, ndarray_labels, smooth_labels, display_results, report_metrics, display_results_dutycycle, smooth_labels_dutycicle


def read_month_data(name,label_opt=0):
    # There are two different columns name format.
    # label_opt=1 specify one of these format.
    if label_opt==1:
        column_interest=['TimeStamp' , 'Pressure High Drive 1  - Pressure [bar]','Pressure Low Drive 1  - Pressure [bar]', 'Actual Speed Drive 1 - Revolutions [rpm]']
        data = pd.read_csv(name, encoding="latin_1",sep=';', usecols=column_interest)
        data.rename(columns = {'TimeStamp':'Timestamp' , 'Pressure High Drive 1  - Pressure [bar]':'High-pressure', 'Pressure Low Drive 1  - Pressure [bar]': 'Low-pressure', 'Actual Speed Drive 1 - Revolutions [rpm]':'Speed'}, inplace = True)
    else:
        column_interest=['time' , 'Pressure High Drive 1','Pressure Low Drive 1', 'Actual Speed Drive 1']
        data = pd.read_csv(name,sep=';', usecols=column_interest)
        data.rename(columns = {'time':'Timestamp' , 'Pressure High Drive 1':'High-pressure', 'Pressure Low Drive 1': 'Low-pressure', 'Actual Speed Drive 1':'Speed'}, inplace = True)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace = True)
    data.reindex()
    return data


# Detection of duty-cicles using heuristic rules
def detect_cycle(value):
    return 'Cycle' if value > 2.5 else 'No_cycle'

def create_reference_label_file(filename,df_name):
    df_label_states = pd.DataFrame()
    df_label_states["start"]=(df_name["start"] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
    df_label_states["end"]=(df_name["end"] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
    df_label_states["label"] = df_name["label"].copy()
    df_label_states.loc[df_label_states["label"]=='E', "label"] = 'B'
    df_label_states.to_csv(filename,index=False,sep="\t",header=False)
    return

def create_segments_state(start_date,end_date,df_dataset):
    complete_range = pd.date_range(start=start_date, end=end_date, freq='1min')
    complete_df = pd.DataFrame({'index': complete_range})

    filtered_df = df_dataset[(df_dataset['index'] >= start_date) & (df_dataset['index'] <= end_date)]

    merged_df = pd.merge(complete_df, filtered_df, on='index', how='left')

    #Compute the start and end of each state
    merged_df['group'] = (merged_df['recognized_label'] != merged_df['recognized_label'].shift()).cumsum()

    dfs = []

    for group_name, group_data in merged_df.groupby('group'):
        label = group_data["recognized_label"].iloc[0]
        if (label != None):
            start = group_data["index"].iloc[0]
            finish = group_data["index"].iloc[-1]
            data_to_append = {'start': start, 'end': finish, 'label': label}
            df = pd.DataFrame(data_to_append,index=[group_name])
            dfs.append(df)

    df_recognized_states = pd.concat(dfs, ignore_index=True)
    return df_recognized_states



def create_segments_cycles(df_recognized_states):
    # Define the normal and abnormal sequences
    normal_sequences = ["BCDCB", "BCDB"]

    # Find indices for normal sequences
    normal_indices = []
    for sequence in normal_sequences:
        normal_indices.extend(find_normal_sequence_indices(df_recognized_states, sequence))

    # find the begin and end of the abnormal cycles
    abnormal_idx_start = []
    df_recognized_states = df_recognized_states.where(pd.notna(df_recognized_states), None)

    for i in range(len(df_recognized_states)-1):
        if df_recognized_states['label'].iloc[i] == 'B' and df_recognized_states['label'].iloc[i+1] == 'C':
            if not any(start <= i < end for start, end in normal_indices):
                abnormal_idx_start.append(i)
        elif df_recognized_states['label'].iloc[i] == 'B' and df_recognized_states['label'].iloc[i+1] == 'D':
            if not any(start <= i < end for start, end in normal_indices):
                abnormal_idx_start.append(i)
        elif df_recognized_states['label'].iloc[i] == 'A' and df_recognized_states['label'].iloc[i+1] == 'C':
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_start.append(i)
        elif df_recognized_states['label'].iloc[i] == 'A' and df_recognized_states['label'].iloc[i+1] == 'D':
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_start.append(i)
        if df_recognized_states['label'].iloc[i] == None and df_recognized_states['label'].iloc[i+1] == 'C':
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_start.append(i)
        elif df_recognized_states['label'].iloc[i] == None and df_recognized_states['label'].iloc[i+1] == 'D':
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_start.append(i)
        elif df_recognized_states['label'].iloc[i] == None and df_recognized_states['label'].iloc[i+1] == 'C':
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_start.append(i)
        elif df_recognized_states['label'].iloc[i] == None and df_recognized_states['label'].iloc[i+1] == 'D':
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_start.append(i)

    abnormal_idx_end = []
    for i in range(len(df_recognized_states)-1):
        if df_recognized_states['label'].iloc[i] == 'C' and df_recognized_states['label'].iloc[i+1] == 'B':
            if not any(start < i <= end for start, end in normal_indices):
                abnormal_idx_end.append(i)
        elif df_recognized_states['label'].iloc[i] == 'D' and df_recognized_states['label'].iloc[i+1] == 'B':
            if not any(start < i <= end for start, end in normal_indices):
                abnormal_idx_end.append(i)
        elif df_recognized_states['label'].iloc[i] == 'C' and df_recognized_states['label'].iloc[i+1] == 'A':
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_end.append(i)
        elif df_recognized_states['label'].iloc[i] == 'D' and df_recognized_states['label'].iloc[i+1] == 'A':
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_end.append(i)
        elif df_recognized_states['label'].iloc[i] == 'C' and df_recognized_states['label'].iloc[i+1] == None:
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_end.append(i)
        elif df_recognized_states['label'].iloc[i] == 'D' and df_recognized_states['label'].iloc[i+1] == None:
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_end.append(i)
        elif df_recognized_states['label'].iloc[i] == 'C' and df_recognized_states['label'].iloc[i+1] == None:
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_end.append(i)
        elif df_recognized_states['label'].iloc[i] == 'D' and df_recognized_states['label'].iloc[i+1] == None:
            if not any(start <= i <= end for start, end in normal_indices):
                abnormal_idx_end.append(i)


    abnormal_cycles_idx = generate_abnormal_cycle(abnormal_idx_start, abnormal_idx_end)


    dfs=[]
    for i in range(0,len(normal_indices)):
        start = df_recognized_states.iloc[normal_indices[i][0]+1,0]
        end = df_recognized_states.iloc[normal_indices[i][1]-1,1]
        data_to_append = {'start': start, 'end': end, 'label': "Normal"}
        df = pd.DataFrame(data_to_append,index=[i])
        dfs.append(df)

    for i in range(0,len(abnormal_cycles_idx)):
        start = df_recognized_states.iloc[abnormal_cycles_idx[i][0]+1,0]
        end = df_recognized_states.iloc[abnormal_cycles_idx[i][1],1]
        data_to_append = {'start': start, 'end': end, 'label': "Abnormal"}
        df = pd.DataFrame(data_to_append,index=[i])
        dfs.append(df)

    df_recognized_cycles = pd.concat(dfs, ignore_index=True)
    
    df_recognized_cycles.sort_values(by='start',inplace=True)
    return df_recognized_cycles

# Function to find the indices of the sequence of normal cycle
def find_normal_sequence_indices(df, sequence):
    indices = []
    for i in range(len(df) - len(sequence) + 1):
        if list(df['label'].iloc[i:i+len(sequence)]) == list(sequence):
            indices.append((i, i+len(sequence)-1))
    return indices

#gived a list of index, the function generate abnormal duty-cycle
def generate_abnormal_cycle(start_index, finish_index):
    segments_boundaries = []
    active_segment = None
    
    # Iterate over the start indexes
    for start in start_index:
        # Find the first finish index greater than or equal to the start index
        finish_candidates = [finish for finish in finish_index if finish >= start]
        
        if finish_candidates:
            # Choose the closest finish index to the start index
            finish = min(finish_candidates)
            if active_segment is None:
                active_segment = (start, finish)
            elif finish <= active_segment[1]:
                active_segment = (active_segment[0], max(active_segment[1], finish))
            else:
                segments_boundaries.append(active_segment)
                active_segment = (start, finish)
    if active_segment is not None:
        segments_boundaries.append(active_segment)
    return segments_boundaries

# Function to find non-consecutive sequences
def find_non_consecutive_sequences(data_df, label_df,delta = pd.Timedelta(minutes=3),column_name='recognized_label'):
    results = []
    for idx, row in label_df.iterrows():
        start_time = row['start']
        end_time = row['end']
        label = row['label']
        
        # Filter data_df based on start and end times
        filtered_df = data_df[(data_df.index >= (start_time-delta)) & (data_df.index <= (end_time+delta))]
        
        # Find non-consecutive sequences in recognized_label
        if not filtered_df.empty:
            filtered_labels = filtered_df[column_name].values
            non_consecutive = [filtered_labels[0]]
            for i in range(1, len(filtered_labels)):
                if filtered_labels[i] != filtered_labels[i - 1]:
                    non_consecutive.append(filtered_labels[i])
                    
            results.append({
                'non_consecutive_labels': non_consecutive,
                'label': label
            })
    return results

# Function to pad sequences to ensure equal length
def pad_sequences(sequences, maxlen, padding_value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            padded_seq = np.concatenate([seq, np.full((maxlen - len(seq)), padding_value)])
        else:
            padded_seq = seq[:maxlen]
        padded_sequences.append(padded_seq)
    return padded_sequences

# function to extract only the [start,end,label] of each detected duty-cycle
def boundaries_cycles(df):
    # Initialize lists to store start and end times
    start_times = []
    end_times = []

    # Flag to track cycle periods
    in_cycle = False

    # Iterate over DataFrame rows
    for time, row in df.iterrows():
        if row["detected_cycles"] == "Cycle":
            if not in_cycle:
                start_times.append(time)
                in_cycle = True
        else:
            if in_cycle:
                end_times.append(prev_time)
                in_cycle = False
        prev_time = time

    # Check if the last row was part of a cycle period
    if in_cycle:
        end_times.append(prev_time)

    # Create new DataFrame with start, end, and label columns
    df_cycle = pd.DataFrame({
        "start": start_times,
        "end": end_times,
        "label": ["Cycle"] * len(start_times)
    })

    return df_cycle

def train_cycle_classifier(classifier,x_train, y_train,seed=42):
    if classifier=="xgboost":
        clf = xgb.XGBClassifier(seed=seed)

        param_grid = {
            'n_estimators': [10,25, 50],  # Number of trees in the forest
            'max_depth': [4,6,8, 10],     # Maximum depth of the tree
            'learning_rate': [0.01, 0.1, 0.2]
    }
    elif classifier=="rf":
        param_grid = {
            'n_estimators': [5,10,25, 50],  # Number of trees in the forest
            'max_depth': [4,6,8,10,15],     # Maximum depth of the tree
        }
        clf = RandomForestClassifier(random_state=seed, criterion="log_loss")

    elif classifier=="dt":
        clf = DecisionTreeClassifier(random_state=seed,criterion='log_loss')

        # Define the parameter grid
        param_grid = {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [5, 10, 20,50],
            'max_features': [None, 'sqrt', 'log2'],
            'criterion': ['gini', 'entropy']
        }

    elif classifier=="nb":
        clf = GaussianNB()

    elif classifier=="xtree":
        clf = ExtraTreesClassifier(random_state=seed,criterion='log_loss')
        param_grid = {
            'n_estimators': [10,25, 50],  # Number of trees in the forest
            'max_depth': [4,6,8, 10],     # Maximum depth of the tree
        }
    elif classifier=="mlp":
        # Define the parameter grid
        param_grid = {
            'hidden_layer_sizes': [(4,), (5,), (6,), (7,),(8,), (9,), (10,), (11,),(12,),(13,), (14,), (15,)],  # Number of neurons in the hidden layer
            'learning_rate_init': [0.001, 0.01, 0.1]            # Learning rates to try
        }
        clf = MLPClassifier(max_iter=3000, random_state=seed)
    else:
        print("clasificador incorrecto")
        return

    if classifier=="nb":
        best_model=clf
    else:
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=8, 
                                   verbose=1,scoring='f1_macro',return_train_score=True)
        grid_search.fit(x_train, y_train)

        # Best parameters from grid search
        # print(f"Best parameters: {grid_search.best_params_}")
        # print(f"Best score: {grid_search.best_score_}")


        # Train the best model on the full training set
        best_model = grid_search.best_estimator_
    best_model.fit(x_train, y_train)
    return best_model


# function to train a specific classifier with hyper-parameters optimization
def train_state_supervised_classifier(classifier,x_train, y_train,seed=42):
    if classifier=="xgboost":
        clf = xgb.XGBClassifier(seed=seed)

        param_grid = {
            'n_estimators': [10,25, 50],  # Number of trees in the forest
            'max_depth': [4,6,8, 10],     # Maximum depth of the tree
            'learning_rate': [0.01, 0.1, 0.2]
    }
    elif classifier=="rf":
        param_grid = {
            'n_estimators': [10,25, 50],  # Number of trees in the forest
            'max_depth': [4,6,8, 10],     # Maximum depth of the tree
        }
        clf = RandomForestClassifier(random_state=seed, criterion="log_loss")

    elif classifier=="dt":
        clf = DecisionTreeClassifier(random_state=seed,criterion='log_loss')

        # Define the parameter grid
        param_grid = {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [5, 10, 20,50],
            'max_features': [None, 'sqrt', 'log2'],
            'criterion': ['gini', 'entropy']
        }

    elif classifier=="nb":
        clf = GaussianNB()

    elif classifier=="xtree":
        clf = ExtraTreesClassifier(random_state=seed,criterion='log_loss')
        param_grid = {
            'n_estimators': [10,25, 50],  # Number of trees in the forest
            'max_depth': [4,6,8, 10],     # Maximum depth of the tree
        }
    elif classifier=="mlp":
        # Define the parameter grid
        param_grid = {
            'hidden_layer_sizes': [(4,), (5,), (6,), (7,),(8,), (9,), (10,), (11,),(12,),(13,), (14,), (15,)],  # Number of neurons in the hidden layer
            'learning_rate_init': [0.001, 0.01, 0.1]            # Learning rates to try
        }
        clf = MLPClassifier(max_iter=1000, random_state=seed)
    else:
        print("clasificador incorrecto")
        return

    if classifier=="nb":
        best_model=clf
    else:
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=5, 
                                   verbose=0,scoring='neg_log_loss',return_train_score=False)
        grid_search.fit(x_train, y_train)

        # Best parameters from grid search
        # print(f"Best parameters: {grid_search.best_params_}")
        # print(f"Best score: {grid_search.best_score_}")


        # Train the best model on the full training set
        best_model = grid_search.best_estimator_
    best_model.fit(x_train, y_train)
    return best_model


def balance_dataset(x_train,y_state):
    le = LabelEncoder()
    le.fit(['A','B','C','D','Z','None'])
    y_train=le.transform(y_state)

    # Oversampling
    val_A,val_B,val_C,val_D = np.count_nonzero(y_train == 0) , np.count_nonzero(y_train == 1) , np.count_nonzero(y_train == 2) , np.count_nonzero(y_train == 3)
    add_A , add_B , add_C , add_D = round(val_A*1) , 0 , round(val_C*1) , round(val_D*0.0)
    strategy = {0:val_A+add_A, 1:val_B+add_B, 2:val_C+add_C, 3:val_D+add_D}
    oversample = SMOTE(sampling_strategy=strategy,k_neighbors=5,random_state=100)
    X_res, y_res = oversample.fit_resample(x_train, y_train)
    # Undersampling
    val_A,val_B,val_C,val_D = np.count_nonzero(y_res == 0) , np.count_nonzero(y_res == 1) , np.count_nonzero(y_res == 2) , np.count_nonzero(y_res == 3)
    del_A , del_B , del_C , del_D = 0 , round(val_B*0.75) , 0 , 0
    strategy = {0:val_A-del_A, 1:val_B-del_B, 2:val_C-del_C, 3:val_D-del_D}
    undersample = RandomUnderSampler(sampling_strategy=strategy,random_state=100)
    X_train_balanced, y_train_balanced = undersample.fit_resample(X_res, y_res)

    return X_train_balanced, y_train_balanced,le


def compute_classification_sedeval(reference_path,result_path,collar = 202.75):
    classes = ['Normal', 'Abnormal']

    # Create a temporary folder to save the replaced labels of the prediction of the algorithms.
    temp_dir = tempfile.mkdtemp()
    filenames = os.listdir(result_path)

    
    # filenames =["jun23_cycle.txt","aug23_cycle.txt","okt23_cycle.txt","dec23_cycle.txt"]
    filenames = [file for file in filenames if file.endswith("_cycle.txt")]


    for file in filenames:
        result_file=os.path.join(result_path, file)
        df = pd.read_csv(result_file,sep="\t",names=["Start","Finish","Label"])
        temp_file = os.path.join(temp_dir, file)
        df.to_csv(temp_file,index=False,sep="\t",header=False)
        df.head()

    file_list = []

    for file in filenames:
        file_list.append({
            'reference_file': os.path.join(reference_path, file),
            'estimated_file': os.path.join(temp_dir, file)
        })

    data = []

    # Get used event labels
    all_data = dcase_util.containers.MetaDataContainer()
    for file_pair in file_list:
        reference_event_list = sed_eval.io.load_event_list(
            filename=file_pair['reference_file']
        )
        estimated_event_list = sed_eval.io.load_event_list(
            filename=file_pair['estimated_file']
        )

        data.append({'reference_event_list': reference_event_list,
                        'estimated_event_list': estimated_event_list})

        all_data += reference_event_list

    # Start evaluating
        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=classes,
            t_collar=collar
        )

        # Go through files
        for file_pair in data:
            event_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )

    # Delete the temporary folder and all its files
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(temp_dir)

    f1_value_abnormal = event_based_metrics.results()["class_wise"]["Abnormal"]['f_measure']['f_measure']
    precision_abnormal =event_based_metrics.results()["class_wise"]["Abnormal"]['f_measure']['precision']
    recall_abnormal = event_based_metrics.results()["class_wise"]["Abnormal"]['f_measure']['recall']

    f1_value_normal = event_based_metrics.results()["class_wise"]["Normal"]['f_measure']['f_measure']
    precision_normal =event_based_metrics.results()["class_wise"]["Normal"]['f_measure']['precision']
    recall_normal = event_based_metrics.results()["class_wise"]["Normal"]['f_measure']['recall']

    f1_value_overall = event_based_metrics.results()["overall"]['f_measure']['f_measure']
    precision_overall =event_based_metrics.results()["overall"]['f_measure']['precision']
    recall_overall = event_based_metrics.results()["overall"]['f_measure']['recall']

    return f1_value_overall,precision_overall,recall_overall,f1_value_abnormal,precision_abnormal,recall_abnormal,f1_value_normal,precision_normal,recall_normal


def replace_detection_labels(df_senial):
   df_senial["Label"].replace("Normal","Cycle",inplace=True)
   df_senial["Label"].replace("Abnormal","Cycle",inplace=True)
   return df_senial
    

def compute_detection_sedeval(reference_path,result_path,collar = 202.75):
    classes = ['Cycle']
    # Create a temporary folder to save the replaced labels of the prediction of the algorithms.
    temp_dir_reference = tempfile.mkdtemp()
    filenames = os.listdir(result_path)
    
    # filenames =["jun23_cycle.txt","aug23_cycle.txt","okt23_cycle.txt","dec23_cycle.txt"]
    filenames = [file for file in filenames if file.endswith("_cycle.txt")]

    for file in filenames:
        result_file=os.path.join(reference_path, file)
        df = pd.read_csv(result_file,sep="\t",names=["Start","Finish","Label"])
        replace_detection_labels(df)
        temp_file = os.path.join(temp_dir_reference, file)
        df.to_csv(temp_file,index=False,sep="\t",header=False)


    # Create a temporary folder to save the replaced labels of the prediction of the algorithms.
    temp_dir_result = tempfile.mkdtemp()

    filenames = [file for file in filenames if file.endswith("_cycle.txt")]


    for file in filenames:
        result_file=os.path.join(result_path, file)
        df = pd.read_csv(result_file,sep="\t",names=["Start","Finish","Label"])
        replace_detection_labels(df)
        temp_file = os.path.join(temp_dir_result, file)
        df.to_csv(temp_file,index=False,sep="\t",header=False)

    file_list = []

    for file in filenames:
        file_list.append({
            'reference_file': os.path.join(temp_dir_reference, file),
            'estimated_file': os.path.join(temp_dir_result, file)
        })

    data = []

    # Get used event labels
    all_data = dcase_util.containers.MetaDataContainer()
    for file_pair in file_list:
        reference_event_list = sed_eval.io.load_event_list(
            filename=file_pair['reference_file']
        )
        estimated_event_list = sed_eval.io.load_event_list(
            filename=file_pair['estimated_file']
        )

        data.append({'reference_event_list': reference_event_list,
                        'estimated_event_list': estimated_event_list})

        all_data += reference_event_list

    # Start evaluating
        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=classes,
            t_collar=collar
        )

        # Go through files
        for file_pair in data:
            event_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )


    # Delete the temporary folder and all its files
    if os.path.exists(temp_dir_reference):
        for root, dirs, files in os.walk(temp_dir_reference, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(temp_dir_reference)

        # Delete the temporary folder and all its files
    if os.path.exists(temp_dir_result):
        for root, dirs, files in os.walk(temp_dir_result, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(temp_dir_result)

    return event_based_metrics.results()["overall"]["f_measure"]["f_measure"]

def extract_features(df):
    #use linear interpolation for the NaN missing values
    df['Speed']=df['Speed'].interpolate("linear")
    df['High-pressure']=df['High-pressure'].interpolate("linear")
    df['Low-pressure']=df['Low-pressure'].interpolate("linear")
        
    df["Speed_order"+str(3)] = df['Speed'].rolling(window=3, center=True, min_periods=1).mean(numeric_only=True)
    df["High-pressure_order"+str(3)] = df['High-pressure'].rolling(window=3, center=True, min_periods=1).mean(numeric_only=True)
    df["Low-pressure_order"+str(3)] = df['Low-pressure'].rolling(window=3, center=True, min_periods=1).mean(numeric_only=True)
    df['Diff-pressure_order'+str(3)]=df['High-pressure_order'+str(3)]-df['Low-pressure_order'+str(3)]

    df["Speed_order"+str(5)] = df['Speed'].rolling(window=5, center=True, min_periods=1).mean(numeric_only=True)
    df["High-pressure_order"+str(5)] = df['High-pressure'].rolling(window=5, center=True, min_periods=1).mean(numeric_only=True)
    df["Low-pressure_order"+str(5)] = df['Low-pressure'].rolling(window=5, center=True, min_periods=1).mean(numeric_only=True)
    df['Diff-pressure_order'+str(5)]=df['High-pressure_order'+str(5)]-df['Low-pressure_order'+str(5)]

    df['Diff-pressure']=df['High-pressure']-df['Low-pressure']

def import_cycle_labels(filename):
    column_interest=['Time(Seconds)' , 'Length(Seconds)',"Label(string)"]
    file_imagimob = pd.read_csv(filename,usecols=column_interest)
    timestamps = df_timestamps(file_imagimob)
    return timestamps

def replace_labels_cycles(df):
    df['label']= df['label'].replace('a', 'Abnormal')
    df['label']= df['label'].replace('n', 'Normal')
    df['label']= df['label'].replace('u/n', 'Normal')
    df['label']= df['label'].replace('u/a', 'Abnormal')

def interpolate_values(df):
    df['Speed']=df['Speed'].interpolate("linear")
    df['High-pressure']=df['High-pressure'].interpolate("linear")
    df['Low-pressure']=df['Low-pressure'].interpolate("linear")