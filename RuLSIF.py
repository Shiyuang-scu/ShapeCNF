from densratio import densratio
from contextlib import contextmanager
import sys, os
import numpy as np;
import gzip;
import pandas as pd
from sklearn import metrics
import torch

def parse_watch_acc(timestamp, path_data):
    os.chdir(path_data)
    file_name = '%s.m_watch_acc.dat' % str(timestamp)
    # open file into text format
    with open(file_name,'r') as fid:
        data = fid.read()
    pass;

    # The original data is made into one whole string object, so I need to split them out one by one.\\
    # That is, replace space and enter. \\
    # Finally, transform them into float.

     # strip space and enter
    data_stripSpaceAndComma = data.replace(' ', ',').split('\n')

     # I found after strip, every 4 emelents become a whole string, so I need to split them out.
    tempe = []
    listtemp = []
    for i in data_stripSpaceAndComma:
        temp=i.split(',')
        listtemp+=temp

     # transform them into float.
    float_data_stripSpaceAndComma = []
    for i in listtemp[:-1]:
        float_data_stripSpaceAndComma.append(float(i))

     # transform array into DataFrame
    length = int(len(float_data_stripSpaceAndComma)/4)
    float_data_stripSpaceAndComma_array = np.reshape(float_data_stripSpaceAndComma, (length, 4))
    float_data_stripSpaceAndComma_dataframe = pd.DataFrame(float_data_stripSpaceAndComma_array, columns=['time','x-axis','y-axis','z-axis'])
      # change the index name to time
    time = float_data_stripSpaceAndComma_dataframe['time'].values.tolist()
    float_data_stripSpaceAndComma_dataframe.index = time
    float_data_stripSpaceAndComma_dataframe = float_data_stripSpaceAndComma_dataframe.drop(columns='time')
    float_data_stripSpaceAndComma_dataframe.index.names = ['Time']

    return float_data_stripSpaceAndComma_dataframe

def parse_labels(uuid, path_label, is_cleaned):
    os.chdir(path_label)
    # read original labels in DataFrame format
    if is_cleaned:
        labels_file_name = '%s.original_labels_of_cleaned_data.csv' % uuid
        with open(labels_file_name,'r') as fid:
            csv_str = pd.read_csv(fid);
    else:
        labels_file_name = '%s.original_labels.csv.gz' % uuid
        with gzip.open(labels_file_name,'r') as fid:
            csv_str = pd.read_csv(fid);
            pass;
    # change the index name to timestamp
    timestamp = csv_str['timestamp'].values.tolist()
    csv_str.index = timestamp
    csv_str.index.names = ['timestamp']
    
    return csv_str

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def evaluate(predicts, y_test):
    Accuracy =  metrics.accuracy_score(y_test, predicts)
    Precision = metrics.precision_score(y_test, predicts)
    F1 = metrics.f1_score(y_test, predicts)
    Recall = metrics.recall_score(y_test, predicts)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicts)
    AUC = metrics.auc(fpr, tpr)

    return {'Accuracy':Accuracy, 'Precision':Precision, 'F1':F1, 'Recall':Recall, 'AUC':AUC}

def get_label(labels_df, timestamp, index):
    current_time = timestamp[index]
    former_time = timestamp[index-1]
    
    foo = (labels_df.loc[current_time].iloc[2:-1] == labels_df.loc[former_time].iloc[2:-1]).all()
    if foo:
        return 0
    else:
        return 1



path_label = '/Users/seunoboru/Desktop/NGNE/CP3106/project data/ExtraSensory/'
path_data = '/Users/seunoboru/Desktop/NGNE/CP3106/project data/watch_acc/'
uuid = 'BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC'

segment_len_list = [10, 20, 50, 100, 500]


# count = 0

for segment_len in segment_len_list:
    print('uuid:{}, segment length:{}'.format(uuid, segment_len))
    PE = []
    y = []
    # count += 1
    # print('**********************************************************')
    # print('Processing %s\'s acc_data, totally processed %d users data' % (uuid, count))

    path_data_of_uuid = path_data + uuid

    print('Loading Data...')
    # Read labels of uuid, and timestamp in this labels data
    labels_df = parse_labels(uuid, path_label, True)[:5000]
    timestamp = labels_df.index.tolist()[:5000]

    print('Performing RuLSIF...')
    # feature extraction
    for i in range(len(timestamp)):
        if (i % 200 == 0):
            print('Totally processed %d timestamp data' % i)
        # Read acc_data
        t_current = timestamp[i]
        watch_acc_df_t_1 = parse_watch_acc(t_current, path_data_of_uuid)
        # get length of acc_data and divide acc_data into several batch, every batch has segment_len points
        len_df = len(watch_acc_df_t_1)
        unit = int(len_df/segment_len)

        # get latter timestamp's acc_data
        t_latter = timestamp[i+1]
        watch_acc_df_t_2 = parse_watch_acc(t_latter, path_data_of_uuid)
        watch_acc_df_t_1and2 = watch_acc_df_t_1.append(watch_acc_df_t_2, ignore_index=True)

        # calculate FastDTW and distance
        # option 2. calculate the last timestamp's acc_data
        if i == len(timestamp)-1:
            pass

        # option 1. calculate the first timestamp's acc_data
        elif i == 0:
            # set the first patch's distance to 0, change point label to 0
            pe_div = 0.0
            change_point = 0.0
            y.append(change_point)
            # calculate distance
            PE.append(pe_div)

            for j in range(1, unit):
                former = watch_acc_df_t_1and2.iloc[(j-1)*segment_len:j*segment_len]
                current = watch_acc_df_t_1and2.iloc[j*segment_len:(j+1)*segment_len]
                with suppress_stdout():
                    pe_div = densratio(former, current).alpha_PE
                y.append(change_point)
                PE.append(pe_div)

        # option 3. calculate usual timestamp's acc_data
        else:
            # get former timestamp's acc_data
            t_former = timestamp[i-1]
            watch_acc_df_t_0 = parse_watch_acc(t_former, path_data_of_uuid)
            former = watch_acc_df_t_0.iloc[-segment_len:]
            current = watch_acc_df_t_1and2.iloc[0:segment_len]
            # calculate distance, set change point label
            with suppress_stdout():
                pe_div = densratio(former, current).alpha_PE
            change_point = get_label(labels_df, timestamp, i)
            y.append(change_point)
            PE.append(pe_div)
            change_point = [0]
            for j in range(1, unit):
                former = watch_acc_df_t_1and2.iloc[(j-1)*segment_len:j*segment_len]
                current = watch_acc_df_t_1and2.iloc[j*segment_len:(j+1)*segment_len]
                with suppress_stdout():
                    pe_div = densratio(former, current).alpha_PE
                y.append(change_point)
                PE.append(pe_div)

    for k in range(len(y)):
        if isinstance(y[k], list):
            y[k] = y[k][0]
    
    eval_metrics = []
    auc = []
    Accuracy = []
    Precision = []
    F1 = []
    Recall = []
    PE = torch.tensor(PE)
    for j in range(int(min(PE)), int(max(PE))):
        predict = (PE >= j).type(torch.FloatTensor)
        per_metrics = evaluate(predict,y)
        eval_metrics.append(per_metrics)
        auc.append(per_metrics['AUC'])
        Accuracy.append(per_metrics['Accuracy'])
        Precision.append(per_metrics['Precision'])
        F1.append(per_metrics['F1'])
        Recall.append(per_metrics['Recall'])
    print('Segment Length: {}'.format(segment_len))
    print(eval_metrics[auc.index(max(auc))])