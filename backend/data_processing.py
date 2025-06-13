import os
import pickle
import numpy as np
import pandas as pd
import pm4py

from typing import List
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from trace_generator import Case, TraceGenerator
from tqdm import tqdm 
from plotting import plot_attributes

from scipy.stats import norm, truncnorm

def load_data(folder_name, file_name):
    file_path = os.path.join('data', folder_name, file_name)
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def save_data(data, folder_name, file_name):
    full_path = os.path.join('data', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_csv_to_df(file_name):
    file_path = os.path.join('raw_data', file_name)
    return pd.read_csv(file_path)

def load_xes_to_df(file_name, folder_name, num_cases=None):
    file_path = os.path.join("data", folder_name, file_name)
    df = pm4py.read_xes(file_path)
    df.rename(columns={'case:concept:name': 'case_id', 'concept:name': 'activity'}, inplace=True)
    columns = ['case_id', 'activity'] + [col for col in df.columns if col not in ['case_id', 'activity']]
    df = df[columns]
    df = df.loc[:, ~df.columns.duplicated()]
    if num_cases:
        df = df[df['case_id'].isin(df['case_id'].unique()[-num_cases:])]
    df = process_df_timestamps(df)
    if folder_name:
        save_data(df, folder_name, "event_log_df.pkl")
    pd.set_option('display.max_columns', None)  # Display all columns
    print(df.head(20))
    return df

def generate_processed_data(process_model, categorical_attributes=[], numerical_attributes=[], num_cases=1000, n_gram=3, folder_name=None):
    print("generating event traces:")
    trace_generator = TraceGenerator(process_model=process_model)
    cases = trace_generator.generate_traces(num_cases=num_cases)
    print("example trace:")
    print(cases[:1])
    print("event pool:")
    print(trace_generator.get_events())
    print("--------------------------------------------------------------------------------------------------")

    print("processing nn data:")
    df = cases_to_dataframe(cases)
    save_data(df, folder_name, "df.pkl")
    
    X, y, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram)
    print("--------------------------------------------------------------------------------------------------")

    return X, y, class_names, feature_names, feature_indices

def create_feature_names(event_pool, attribute_pools, numerical_attributes, n_gram):
    feature_names = []
    # Add event features for each n-gram step
    for index in range(n_gram, 0, -1):
        for event in sorted(event_pool):
            feature_names.append(f"-{index}. Event = {event}")
    # Add categorical attribute features
    for attribute_name, possible_values in sorted(attribute_pools.items()):
        for value in sorted(possible_values):
            feature_names.append(f"{attribute_name} = {value}")
    # Add numerical attribute features
    for numerical_attr in sorted(numerical_attributes):
        feature_names.append(numerical_attr)
    return feature_names

def create_feature_indices(event_pool, attribute_pools, numerical_attributes, n_gram):
    num_events = len(sorted(event_pool))  # Sorted event pool
    feature_indices = {}
    # Allocate indices for events
    index = num_events * n_gram
    # Allocate indices for categorical attributes
    for attribute_name, possible_values in sorted(attribute_pools.items()):
        num_values = len(sorted(possible_values))
        feature_indices[attribute_name] = list(range(index, index + num_values))
        index += num_values
    # Allocate indices for numerical attributes
    for numerical_attr in sorted(numerical_attributes):
        feature_indices[numerical_attr] = [index]
        index += 1
    return feature_indices
    
def create_attribute_pools(df, case_attributes):
    attribute_pools = {}
    for attr in case_attributes:
        if attr in df.columns:
            attribute_pools[attr] = sorted(df[attr].dropna().unique().tolist())  # Ensure sorted order
        else:
            raise KeyError(f"Attribute '{attr}' is not in the DataFrame.")
    return attribute_pools 

def process_df_timestamps(df):
    if "time" in df.columns:
        time_column = "time"
    elif "time:timestamp" in df.columns:
        time_column = "time:timestamp"
    else:
        return df

    print(time_column)
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=['case_id', time_column]).reset_index(drop=True)
    df['time_delta'] = df.groupby('case_id')[time_column].diff().dt.total_seconds()
    df['time_delta'] = df['time_delta'].fillna(0)  # Set time_delta to 0 for the first event in each case
    df['time_of_day'] = df[time_column].dt.hour / 24 + df[time_column].dt.minute / 1440 + df[time_column].dt.second / 86400
    df['day_of_week'] = df[time_column].dt.dayofweek  # Monday=0, Sunday=6
    return df

def cases_to_dataframe(cases: List[Case]) -> pd.DataFrame:
    """
    Converts a list of Case objects into a pandas DataFrame with columns:
    'case_id', 'activity', and one column for each case attribute (categorical and numerical).
    """
    rows = []
    for case in cases:
        for event in case.events:
            row = {
                'case_id': case.case_id,
                'activity': event.activity,
                'time': event.timestamp
            }
            row.update(case.categorical_attributes)
            row.update(case.numerical_attributes)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df_to_xes(df)
    df = process_df_timestamps(df)
    return df

def df_to_xes(df: pd.DataFrame):
    import pandas as pd
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    """
    Converts a pandas DataFrame to an XES file format and saves it.
    """
    log = EventLog()

    # Get the list of extra columns excluding the standard ones
    standard_cols = {'case_id', 'activity', 'time'}
    extra_cols = [col for col in df.columns if col not in standard_cols]

    # Group by case_id to create traces
    for case_id, group in df.groupby('case_id'):
        trace = Trace()
        trace.attributes["concept:name"] = case_id
        
        # Sort events by time within the case
        group = group.sort_values(by='time')
        
        for _, row in group.iterrows():
            event = Event()
            event["concept:name"] = row['activity']
            event["time:timestamp"] = row['time']
            
            # Add additional categorical and numerical attributes to event
            for col in extra_cols:
                event[col] = row[col]
            
            trace.append(event)
            
        log.append(trace)

    # Export to XES file
    xes_exporter.apply(log, "cancer_screening.xes")
    print("XES log saved to 'output_log.xes'")

def process_df(df, categorical_attributes, numerical_attributes, n_gram=3):
    """Processes dataframe data for neural network training"""
    # keep only specified attributes
    standard_attributes = ['case_id', 'activity']
    categorical_attributes.sort()
    numerical_attributes.sort()
    attributes_to_include = standard_attributes + categorical_attributes + numerical_attributes 
    print(attributes_to_include)
    df = df[attributes_to_include]

    # create the meta-information
    class_names = sorted(df["activity"].unique().tolist() + ["<PAD>"])
    attribute_pools = create_attribute_pools(df, categorical_attributes)
    feature_names = create_feature_names(class_names, attribute_pools, numerical_attributes, n_gram)
    feature_indices = create_feature_indices(class_names, attribute_pools, numerical_attributes, n_gram)
    print(f"class_names: {class_names}")
    print(f"amount of classes = {len(class_names)}, ngram = {n_gram}")
    print(f"attribute_pools: {attribute_pools}")
    print(f"feature_names: {feature_names}")
    print(f"feature_indices: {feature_indices}")

    # one-hot encode activities
    activity_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[class_names])
    activity_ohe = activity_encoder.fit_transform(df[['activity']])
    pad_activity_idx = activity_encoder.categories_[0].tolist().index("<PAD>")
    
    # one-hot encode categorical case attributes dynamically
    attribute_encoders = {}
    attributes_ohe_dict = {}
    for attr in categorical_attributes:
        print(attr)
        print(attribute_pools[attr])
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[attribute_pools[attr]])
        attributes_ohe_dict[attr] = encoder.fit_transform(df[[attr]])
        attribute_encoders[attr] = encoder

    # Scale numerical attributes between 0 and 1 based on training data's min/max values
    numerical_scalers = {}
    for attr in numerical_attributes:
        scaler = MinMaxScaler()
        scaler.fit(df[[attr]])
        numerical_scalers[attr] = scaler

    # group by case_id and create sequences
    grouped = df.groupby('case_id')
    cases = []
    for case_id, group in tqdm(grouped, desc="preparing cases"):
        activities = activity_encoder.transform(group[['activity']])
        attributes = {attr: attribute_encoders[attr].transform(group[[attr]]) for attr in categorical_attributes}
        # Scale numerical attributes within the group
        for attr in numerical_attributes:
            group[attr] = numerical_scalers[attr].transform(group[[attr]])
        cases.append((activities, attributes, group[numerical_attributes].values))
    pd.set_option('display.max_columns', None)  # Display all columns
    print(grouped.head(20))

    # Generate n-grams with padding
    X, y = [], []
    pad_activity = activity_encoder.transform([["<PAD>"]])  # Get one-hot for <PAD>
    pad_attributes = {attr: np.zeros((1, attributes_ohe_dict[attr].shape[1])) for attr in categorical_attributes}
    pad_numerical = np.zeros((1, len(numerical_attributes)))  # Padding for numerical features

    for activities, attributes, numerical in tqdm(cases, desc="encoding cases"):
        # Pad activities
        padded_activities = np.vstack([pad_activity] * n_gram + [activities])
        
        # Pad categorical attributes
        padded_attributes = {attr: np.vstack([pad_attributes[attr]] * n_gram + [attributes[attr]])
                            for attr in sorted(attributes)}

        # Pad numerical attributes
        padded_numerical = np.vstack([pad_numerical] * n_gram + [numerical])

        for i in range(len(activities)):  # Start from 0 and include all real activities
            x_activities = padded_activities[i:i + n_gram]
            if categorical_attributes:
                x_attributes = np.hstack([
                    padded_attributes[attr][i + n_gram] 
                    for attr in categorical_attributes
                ])
            else:
                x_attributes = np.array([])

            if numerical_attributes:
                x_numerical = padded_numerical[i + n_gram]
            else:
                x_numerical = np.array([])

            x_combined = np.hstack([x_activities.flatten(), x_attributes, x_numerical])  # Combine activities, attributes, and numerical features
            
            y_next_activity = activities[i]  # Predict the actual next activity
            X.append(x_combined)
            y.append(y_next_activity)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    n = 10
    print("example nn inputs:")
    print(X[:n])
    print("example nn outputs:")
    print(y[:n])

    return X, y, class_names, feature_names, feature_indices

def enrich_df(df: pd.DataFrame, rules: list, folder_name: str):
    def generate_value(distribution, rng):
        """Generates a value based on the specified distribution."""
        if distribution['type'] == 'discrete':
            values, weights = zip(*distribution['values'])
            return rng.choice(values, p=np.array(weights) / sum(weights))
        elif distribution['type'] == 'normal':
            mean, std = distribution['mean'], distribution['std']
            a, b = distribution.get('min', -np.inf), distribution.get('max', np.inf)

            if np.isinf(a) and np.isinf(b):
                return norm.rvs(loc=mean, scale=std, random_state=rng)
            else:
                # Scale the bounds relative to the mean and standard deviation
                a, b = (a - mean) / std, (b - mean) / std
                return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng)
        else:
            raise ValueError("Unsupported distribution type.")

    rng = np.random.default_rng()

    # Identify unique cases in the log
    case_ids = df['case_id'].unique()

    # Initialize a dictionary to hold the generated attributes for each case
    case_attributes = {rule['attribute']: {} for rule in rules}

    for case_id in tqdm(case_ids, desc="enriching cases"):
        # Extract events for the current case
        case_events = df[df['case_id'] == case_id]['activity'].tolist()

        for rule in rules:
            subsequence = rule['subsequence']
            attribute = rule['attribute']
            distribution = rule['distribution']

            # Check if the subsequence is present in the case's events
            if any(
                case_events[i:i + len(subsequence)] == subsequence
                for i in range(len(case_events) - len(subsequence) + 1)
            ):
                # Generate a value based on the distribution
                case_attributes[attribute][case_id] = generate_value(distribution, rng)
                #print(f"Matched: {subsequence} to {case_attributes[attribute][case_id]}")

    # Add generated attributes as new columns to the DataFrame
    for attribute, values in case_attributes.items():
        df[attribute] = df['case_id'].map(values)
    
    # plot these for evaluation of the success
    #plot_attributes(df, rules, folder_name)

    return df


def mine_bpm(file_name, folder_name):
    print("Mining BPM...")
    pd.set_option('display.max_columns', None)
    file_path = os.path.join("raw_data", file_name)
    df = pm4py.read_xes(file_path)
    if "bpi" in folder_name:
        df = df[df['concept:name'].str.startswith('A_')]
    output_dir = os.path.join("img", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # the log is filtered on the top 5 variants
    variants = [5, 10, 15, 20]
    for variant in variants:
        data_df = pm4py.filter_variants_top_k(df, variant)
        dfg, start_activities, end_activities = pm4py.discover_dfg(data_df)
        pm4py.save_vis_dfg(dfg, start_activities, end_activities, os.path.join(output_dir, f"dfg_{variant}.png"), format='png')
    print("--------------------------------------------------------------------------------------------------")


def k_fold_cross_validation(df, categorical_attributes, numerical_attributes, critical_decisions, n_gram=3, k=10):
    # Ensure the attributes are sorted consistently
    categorical_attributes.sort()
    numerical_attributes.sort()
    numerical_thresholds = {}

    # Group by case_id for consistent splitting
    grouped = df.groupby('case_id')
    case_ids = list(grouped.groups.keys())
    
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(case_ids)):
        print(f"Processing fold {fold + 1}/{k}")
        
        # Split the data
        train_case_ids = [list(case_ids)[i] for i in train_idx]
        test_case_ids = [list(case_ids)[i] for i in test_idx]
        
        train_df = df[df['case_id'].isin(train_case_ids)]
        test_df = df[df['case_id'].isin(test_case_ids)]
        
        # Process training data for scaling and encoding
        class_names = sorted(train_df["activity"].unique().tolist() + ["<PAD>"])
        activity_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[class_names])
        activity_encoder.fit(train_df[['activity']])
        
        attribute_encoders = {}
        for attr in categorical_attributes:
            attribute_pools = sorted(train_df[attr].unique())
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[attribute_pools])
            encoder.fit(train_df[[attr]])
            attribute_encoders[attr] = encoder
        
        numerical_scalers = {}
        for attr in numerical_attributes:
            scaler = MinMaxScaler()
            scaler.fit(train_df[[attr]])
            numerical_scalers[attr] = scaler
        
        # Process training and test data into n-grams
        X_train, y_train = transform_samples(train_df, activity_encoder, attribute_encoders, numerical_scalers,
                                         categorical_attributes, numerical_attributes, n_gram)
        X_test, y_test = transform_samples(test_df, activity_encoder, attribute_encoders, numerical_scalers,
                                       categorical_attributes, numerical_attributes, n_gram, train=False)

        # get the thresholds for the numerical attributes
        for decision in critical_decisions:
            if len(decision.attributes) == 1:
                attribute = decision.attributes[0]
                if attribute in numerical_attributes:
                    threshold = numerical_scalers[attribute].transform([[decision.threshold]])
                    numerical_thresholds[attribute] = threshold[0][0]

        

        print(f"Fold {fold + 1}: Train samples = {len(X_train)}, Test samples = {len(X_test)}")
        print(numerical_thresholds)
        yield X_train, y_train, X_test, y_test, numerical_thresholds


def transform_samples(df, activity_encoder, attribute_encoders, numerical_scalers,
                  categorical_attributes, numerical_attributes, n_gram, train=True):
    """
    Generate n-gram sequences from the dataset.
    """
    grouped = df.groupby('case_id')
    cases = []

    # Prepare data for each case
    for case_id, group in tqdm(grouped, desc="Preparing cases" if train else "Processing test cases"):
        activities = activity_encoder.transform(group[['activity']])
        attributes = {attr: attribute_encoders[attr].transform(group[[attr]]) for attr in categorical_attributes}
        
        # Scale numerical attributes
        for attr in numerical_attributes:
            group[attr] = numerical_scalers[attr].transform(group[[attr]])
        cases.append((activities, attributes, group[numerical_attributes].values))
    
    # Generate n-grams with padding
    X, y = [], []
    pad_activity = activity_encoder.transform([["<PAD>"]])
    pad_attributes = {attr: np.zeros((1, enc.categories_[0].shape[0])) for attr, enc in attribute_encoders.items()}
    pad_numerical = np.zeros((1, len(numerical_attributes)))

    for activities, attributes, numerical in cases:
        padded_activities = np.vstack([pad_activity] * n_gram + [activities])
        padded_attributes = {attr: np.vstack([pad_attributes[attr]] * n_gram + [attributes[attr]])
                             for attr in sorted(attributes)}
        padded_numerical = np.vstack([pad_numerical] * n_gram + [numerical])

        for i in range(len(activities)):
            x_activities = padded_activities[i:i + n_gram]
            x_attributes = np.hstack([padded_attributes[attr][i + n_gram] for attr in categorical_attributes])
            x_numerical = padded_numerical[i + n_gram]

            x_combined = np.hstack([x_activities.flatten(), x_attributes, x_numerical])
            y_next_activity = activities[i]
            
            X.append(x_combined)
            y.append(y_next_activity)
    
    return np.array(X), np.array(y)
