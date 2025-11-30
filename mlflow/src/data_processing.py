import pandas as pd
import numpy as np

def load_and_process_data(data_dir="data"):
    print(" Loading data...")
    # Load DataFrames
    telemetry = pd.read_csv(f'{data_dir}/PdM_telemetry.csv')
    errors = pd.read_csv(f'{data_dir}/PdM_errors.csv')
    maint = pd.read_csv(f'{data_dir}/PdM_maint.csv')
    failures = pd.read_csv(f'{data_dir}/PdM_failures.csv')
    machines = pd.read_csv(f'{data_dir}/PdM_machines.csv')

    # Formatting Datetimes
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")
    errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
    maint['datetime'] = pd.to_datetime(maint['datetime'], format="%Y-%m-%d %H:%M:%S")
    failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
    machines['model'] = machines['model'].astype('object')

    print(" Feature Engineering: Telemetry (This takes time)...")
    # Telemetry Lag Features (Rolling 3h and 24h)
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    
    # Calculate 3H Mean
    for col in fields:
        temp.append(pd.pivot_table(telemetry, index='datetime', columns='machineID', values=col)
                    .resample('3h', closed='left', label='right').mean().unstack())
    telemetry_mean_3h = pd.concat(temp, axis=1)
    telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
    telemetry_mean_3h.reset_index(inplace=True)

    # Calculate 3H Std
    temp = []
    for col in fields:
        temp.append(pd.pivot_table(telemetry, index='datetime', columns='machineID', values=col)
                    .resample('3h', closed='left', label='right').std().unstack())
    telemetry_sd_3h = pd.concat(temp, axis=1)
    telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
    telemetry_sd_3h.reset_index(inplace=True)

    # Calculate 24H Mean
    temp = []
    for col in fields:
        temp.append(pd.pivot_table(telemetry, index='datetime', columns='machineID', values=col)
                    .resample('3h', closed='left', label='right').first().unstack().rolling(window=24, center=False).mean())
    telemetry_mean_24h = pd.concat(temp, axis=1)
    telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
    telemetry_mean_24h.reset_index(inplace=True)
    telemetry_mean_24h = telemetry_mean_24h.dropna()

    # Calculate 24H Std
    temp = []
    for col in fields:
        temp.append(pd.pivot_table(telemetry, index='datetime', columns='machineID', values=col)
                    .resample('3h', closed='left', label='right').first().unstack().rolling(window=24, center=False).std())
    telemetry_sd_24h = pd.concat(temp, axis=1)
    telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
    telemetry_sd_24h.reset_index(inplace=True)
    telemetry_sd_24h = telemetry_sd_24h.dropna()

    # Merge Telemetry Features
    telemetry_feat = pd.concat([telemetry_mean_3h,
                                telemetry_sd_3h.iloc[:, 2:6],
                                telemetry_mean_24h.iloc[:, 2:6],
                                telemetry_sd_24h.iloc[:, 2:6]], axis=1).dropna()

    print(" Feature Engineering: Errors...")
    error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
    error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
    error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)
    
    temp = []
    fields = ['error%d' % i for i in range(1, 6)]
    for col in fields:
        temp.append(pd.pivot_table(error_count, index='datetime', columns='machineID', values=col)
                    .resample('3h', closed='left', label='right').first().unstack().rolling(window=24, center=False).sum())
    error_count = pd.concat(temp, axis=1)
    error_count.columns = [i + 'count' for i in fields]
    error_count.reset_index(inplace=True)
    error_count = error_count.dropna()

    print(" Feature Engineering: Maintenance & Machines...")
    # Maintenance days since last replacement
    comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
    comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']
    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep, on=['datetime', 'machineID'], how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])
    
    components = ['comp1', 'comp2', 'comp3', 'comp4']
    for comp in components:
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
        comp_rep[comp] = comp_rep[comp].ffill()
    
    comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]
    for comp in components:
        comp_rep[comp] = (comp_rep["datetime"] - pd.to_datetime(comp_rep[comp])) / np.timedelta64(1, "D")

    # Final Merge
    final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(machines, on=['machineID'], how='left')

    # Labeling
    labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
    labeled_features['failure'] = labeled_features['failure'].fillna(method='bfill', limit=7) # Look ahead 24h
    labeled_features['failure'] = labeled_features['failure'].fillna('none')

    # One Hot Encode 'model' column specifically
    labeled_features = pd.get_dummies(labeled_features, columns=['model'])

    print(f" Data Processed. Shape: {labeled_features.shape}")
    return labeled_features