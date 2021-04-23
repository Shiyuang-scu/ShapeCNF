import numpy as np;
import os
import pandas as pd
import gzip
from sklearn import metrics
import torch 
from torch import nn
from sklearn.model_selection import KFold
from dtaidistance import dtw
import matplotlib.pyplot as plt




def parse_watch_acc(timestamp, path_data):
    ''' Parse the accelerometer data into dataframe

    args: 
        timestamp: int, specify the watch_acc.dat
        path_data: path of the acc_data
    return:
        float_data_stripSpaceAndComma_dataframe: dataframe of the accelerometer data
    '''


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



def to_array(watch_acc_df):
    ''' Convert dataframe to array in order to fastdtw

    args: 
        watch_acc_df: dataframe of the accelerometer data
    return:
        watch_acc_df_xaxis: array of the x axis accelerometer data
        watch_acc_df_yaxis: array of the y axis accelerometer data
        watch_acc_df_zaxis: array of the z axis accelerometer data
    '''


    watch_acc_df_xaxis = np.array(watch_acc_df['x-axis'], dtype=np.double)
    watch_acc_df_yaxis = np.array(watch_acc_df['y-axis'], dtype=np.double)
    watch_acc_df_zaxis = np.array(watch_acc_df['z-axis'], dtype=np.double)
    return watch_acc_df_xaxis, watch_acc_df_yaxis, watch_acc_df_zaxis




def feature_extraction(watch_acc_df_former, watch_acc_df, watch_acc_df_latter, psi):
    ''' deature extraction using fastdtw

    args: 
        watch_acc_df_former: former data
        watch_acc_df: current data that we need to extract features
        watch_acc_df_latter: latter data
        psi: relaxation length
    return:
        D: the shape-based features
    '''


    try:
        # make df to array
        former_array_xaxis, former_array_yaxis, former_array_zaxis = to_array(watch_acc_df_former)
        array_xaxis, array_yaxis, array_zaxis = to_array(watch_acc_df)
        latter_array_xaxis, latter_array_yaxis, latter_array_zaxis = to_array(watch_acc_df_latter)
        # get distance between data and its former data
        distance_former_xaxis = dtw.distance(former_array_xaxis, array_xaxis, psi=psi)
        distance_former_yaxis = dtw.distance(former_array_yaxis, array_yaxis, psi=psi)
        distance_former_zaxis = dtw.distance(former_array_zaxis, array_zaxis, psi=psi)
        # get distance between data and its latter data
        distance_latter_xaxis = dtw.distance(latter_array_xaxis, array_xaxis, psi=psi)
        distance_latter_yaxis = dtw.distance(latter_array_yaxis, array_yaxis, psi=psi)
        distance_latter_zaxis = dtw.distance(latter_array_zaxis, array_zaxis, psi=psi)
    except Exception as e:
            print("Oops!", e.__class__, "occurred in func weather.")

    D = [(distance_former_xaxis-distance_latter_xaxis), (distance_former_yaxis-distance_latter_yaxis),\
        (distance_former_zaxis-distance_latter_zaxis)]
    return D




def get_label(labels_df, timestamp, index):
    ''' Convert activity labels into change-point labels

    args: 
        labels_df: dataframe which contains activity labels
        timestamp: list of timestamps
        index: specify timestamp
    return:
        0
        1
    '''



    current_time = timestamp[index]
    former_time = timestamp[index-1]
    
    foo = (labels_df.loc[current_time].iloc[2:-1] == labels_df.loc[former_time].iloc[2:-1]).all()
    if foo:
        return [0]
    else:
        return [1]





def read_uuid(path_of_uuid):
    ''' Get the uuid list

    args: 
        path_of_uuid: path of the uuid file
    return:
        list_of_uuid: list of uuids
    '''


    list_of_uuid = []
    os.chdir(path_of_uuid)
    
    for file in os.listdir(path_of_uuid):
        with open(file, 'r') as my_file:
            temp = my_file.read().splitlines() 
            list_of_uuid += temp
    list_of_uuid = list(set(list_of_uuid))
    
    return list_of_uuid




def parse_labels(uuid, path_label, is_cleaned):
    ''' Parse activity labels

    args: 
        uuid: specify the uuid
        path_label: path of the label file
        is_cleaned: boolean
    return:
        csv_str: dataframe of labels
    '''




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



def load_file(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe.iloc[:,1:5]



def load_group(timestamp, segment_len, psi):
    ''' Convert one timestamp dataframe to data (x) and label (y)

    '''


    x = []
    y = []

    for time in timestamp:
        extracted_feature_filename = '%s.extracted_feature.segment%d.psi%d.csv' % (time,segment_len,psi)
        data = load_file(extracted_feature_filename)
        x.append(data.iloc[:,0:-1].values)
        y.append(data.iloc[:,-1].values)

	# # stack group so that features are the 3rd dimension
    x = np.concatenate(x)
    y = np.concatenate(y)

    return x, y




def load_dataset_group(uuid_group, path_data, path_label, segment_len, psi):
    ''' Convert all dataframes, which are from uuid user, to data and label
    
    '''

    os.chdir(path_data)
    x = []
    y = []

    for uuid in uuid_group:
        labels_df = parse_labels(uuid, path_label, True)
        os.chdir(path_data + uuid)
        # timestamp = labels_df.index.tolist()[:2000]
        timestamp = labels_df.index.tolist()
        group_data, group_label = load_group(timestamp,segment_len,psi)
        x.append(group_data)
        y.append(group_label)

    x = np.concatenate(x)
    y = np.concatenate(y)

    return x, y



def under_sampling(x, y):
    ''' Under sampling for imbalanced dataset
    
    '''


    index_1 = [idx for idx, val in enumerate(y) if val > 0]

    # take former 2 data points and 2 latter data points: [former1, former2, current, latter1, latter2]
    index_neighbor = [[idx-2,idx-1,idx+1,idx+2] for idx in index_1]
    index_neighbor = sum(index_neighbor,[])
    index = sorted(index_neighbor+index_1)

    return x[index], y[index]



def evaluate(batchs_test, y_test, cnf, stride_size):
    ''' Evaluate the performance of ShapeCNF
    
    '''

    output_list = []
    for batch in batchs_test[:-1]:
        output_p = cnf(batch)
        output_p = torch.squeeze(output_p)[0:stride_size]
        output_list.append(output_p)

    output_p = cnf(batch)
    output_p = torch.squeeze(output_p)
    output_list.append(output_p)

    output_pre = torch.reshape(torch.stack(output_list[:-1]), (-1,))
    output_pre = torch.cat((output_pre, output_list[-1]),0)

    probs = output_pre.sigmoid()
    probs_ = probs.type(torch.FloatTensor).detach().numpy()
    predicts = (probs >= 0.5).type(torch.FloatTensor)

    Accuracy =  metrics.accuracy_score(y_test[:len(predicts)], predicts)
    Precision = metrics.precision_score(y_test[:len(predicts)], predicts)
    # Precision = metrics.precision_score(y_test[:len(predicts)], predicts, labels=np.unique(predicts))
    F1 = metrics.f1_score(y_test[:len(predicts)], predicts)
    # F1 = metrics.f1_score(y_test[:len(predicts)], predicts, labels=np.unique(predicts))
    Recall = metrics.recall_score(y_test[:len(predicts)], predicts)
    fpr, tpr, thresholds = metrics.roc_curve(y_test[:len(probs_)], probs_)
    AUC = metrics.auc(fpr, tpr)

    # if AUC >= 0.88:
    #     print(list(fpr), list(tpr))
    #     plt.figure(figsize=(7,7))
    #     plt.title('Receiver Operating Characteristic')
    #     plt.plot(fpr, tpr, color='red', marker='p',  linewidth=2, markersize=7, label = 'ShapeCNF (AUC = %0.4f)' % AUC)
    #     plt.legend(loc = 'lower right')
    #     # plt.plot([0, 1], [0, 1],'r--')
    #     plt.xlim([0, 1])
    #     plt.ylim([0, 1])
    #     plt.ylabel('True Positive Rate')
    #     plt.xlabel('False Positive Rate')
    #     plt.show()
    # # else:
    # #     print(fpr, tpr)
    # #     print(AUC)

    return {'Accuracy':Accuracy, 'Precision':Precision, 'F1':F1, 'Recall':Recall, 'AUC':AUC}



def window_split(x, window_size=5, stride=3, keep_short_tails=False):
    ''' split data into batches

    '''


    length = x.size(0)
    splits = []

    if keep_short_tails:
        for slice_start in range(0, length, stride):
            slice_end = min(length, slice_start + window_size)
            splits.append(x[slice_start:slice_end])
    else:
        for slice_start in range(0, length - window_size + 1, stride):
            slice_end = slice_start + window_size
            splits.append(x[slice_start:slice_end])

    return splits



def DTW(segment_len, path_label, path_data, uuid_list, psi):
    ''' Performing DTW

    '''


    count = 0
    for uuid in uuid_list:
        count += 1
        print('**********************************************************')
        print('Processing %s\'s acc_data, totally processed %d users data' % (uuid, count))

        path_data_of_uuid = path_data + uuid
        
        # Read labels of uuid, and timestamp in this labels data
        labels_df = parse_labels(uuid, path_label, True)
        timestamp = labels_df.index.tolist()[:2000]
        # timestamp = labels_df.index.tolist()

        # feature extraction
        for i in range(len(timestamp)):
            if (i % 1000 == 0):
                print('Totally processed %d timestamp data' % i)

            extracted_feature = []

            # Read acc_data
            t_current = timestamp[i]
            watch_acc_df_t_1 = parse_watch_acc(t_current, path_data_of_uuid)

            # get length of acc_data and divide acc_data into several batch, every batch has segment_len points
            len_df = len(watch_acc_df_t_1)
            unit = int(len_df/segment_len)

            # calculate FastDTW and distance
            # option 1. calculate the first timestamp's acc_data
            if i == 0:
                # get latter timstamp's acc_data
                t_latter = timestamp[i+1]
                watch_acc_df_t_2 = parse_watch_acc(t_latter, path_data_of_uuid)
                watch_acc_df_t_1and2 = watch_acc_df_t_1.append(watch_acc_df_t_2, ignore_index=True)

                # set the first patch's distance to 0, change point label to 0
                distance = [0.0, 0.0, 0.0]
                change_point = [0.0]
                feature = distance + change_point

                # calculate distance
                extracted_feature.append(feature)
                for j in range(1, unit):
                    former = watch_acc_df_t_1and2.iloc[(j-1)*segment_len:j*segment_len]
                    current = watch_acc_df_t_1and2.iloc[j*segment_len:(j+1)*segment_len]
                    latter = watch_acc_df_t_1and2.iloc[(j+1)*segment_len:(j+2)*segment_len]
                    distance = feature_extraction(former, current, latter, psi)
                    feature = distance + change_point
                    extracted_feature.append(feature)
            
            # option 2. calculate the last timestamp's acc_data
            elif i == len(timestamp)-1:

                # get former timestamp's acc_data
                t_former = timestamp[i-1]
                watch_acc_df_t_0 = parse_watch_acc(t_former, path_data_of_uuid)
                former = watch_acc_df_t_0.iloc[-segment_len:]
                current = watch_acc_df_t_1.iloc[0:segment_len]
                latter = watch_acc_df_t_1.iloc[segment_len:2*segment_len]

                # calculate distance, set change point label
                distance = feature_extraction(former, current, latter, psi)
                change_point = get_label(labels_df, timestamp, i)
                feature = distance + change_point
                extracted_feature.append(feature)
                change_point = [0]
                for j in range(1, unit-1):
                    former = watch_acc_df_t_1.iloc[(j-1)*segment_len:j*segment_len]
                    current = watch_acc_df_t_1.iloc[j*segment_len:(j+1)*segment_len]
                    latter = watch_acc_df_t_1.iloc[(j+1)*segment_len:(j+2)*segment_len]
                    distance = feature_extraction(former, current, latter, psi)
                    feature = distance + change_point
                    extracted_feature.append(feature)

                # set the last patch's distance to 0
                distance = [0.0, 0.0, 0.0]
                feature = distance + change_point
                extracted_feature.append(feature)
            
            # option 3. calculate usual timestamp's acc_data
            else:
                # get latter timestamp's acc_data
                t_latter = timestamp[i+1]
                watch_acc_df_t_2 = parse_watch_acc(t_latter, path_data_of_uuid)
                watch_acc_df_t_1and2 = watch_acc_df_t_1.append(watch_acc_df_t_2, ignore_index=True)

                # get former timestamp's acc_data
                t_former = timestamp[i-1]
                watch_acc_df_t_0 = parse_watch_acc(t_former, path_data_of_uuid)
                former = watch_acc_df_t_0.iloc[-segment_len:]
                current = watch_acc_df_t_1and2.iloc[0:segment_len]
                latter = watch_acc_df_t_1and2.iloc[segment_len:2*segment_len]

                # calculate distance, set change point label
                distance = feature_extraction(former, current, latter, psi)
                change_point = get_label(labels_df, timestamp, i)
                feature = distance + change_point
                extracted_feature.append(feature)
                change_point = [0]
                for j in range(1, unit):
                    former = watch_acc_df_t_1and2.iloc[(j-1)*segment_len:j*segment_len]
                    current = watch_acc_df_t_1and2.iloc[j*segment_len:(j+1)*segment_len]
                    latter = watch_acc_df_t_1and2.iloc[(j+1)*segment_len:(j+2)*segment_len]
                    distance = feature_extraction(former, current, latter, psi)
                    feature = distance + change_point
                    extracted_feature.append(feature)
            
            # convert feature list to feature dataframe
            extracted_feature_df = pd.DataFrame(extracted_feature, columns=['x-axis', 'y-axis', 'z-axis', 'label'])
            
            # Save feature dataframe to csv format
            os.chdir(path_data_of_uuid)
            extracted_feature_filename = '%s.extracted_feature.segment%d.psi%d.csv' % (timestamp[i],segment_len,psi)
            extracted_feature_df.to_csv(extracted_feature_filename)



def draw_graph(loss_list, metric):
    ''' Draw the loss trend figure

    '''


    plt.figure(figsize=(20,4))
    plt.plot(list(range(len(loss_list))),loss_list)
    plt.xticks(list(range(0,len(loss_list)+1,10)))

    plt.xlabel('Epochs')
    plt.ylabel(metric)

    filename = '/Users/seunoboru/Desktop/NGNE/CP3106/project data/figure/{}.trend.png'.format(metric)
    plt.savefig(filename, dpi=300)



class cnf(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, iteration=10):
        super(cnf, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.iteration = iteration
        self.W = nn.Parameter(torch.rand(num_nodes, num_nodes))
        
        if isinstance(hidden_size,list):
            self.ClassifierLayer = nn.Sequential(
                        nn.Linear(self.input_size, self.hidden_size[0]),
                        nn.BatchNorm1d(hidden_size[0]),
                        nn.LeakyReLU(),
                        nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                        nn.BatchNorm1d(hidden_size[1]),
                        nn.LeakyReLU()
                    )
            self.fc = nn.Linear(self.hidden_size[1], 1)
        else:
            self.ClassifierLayer = nn.Sequential(
                        nn.Linear(self.input_size, self.hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.LeakyReLU(),
                    )
            self.fc = nn.Linear(self.hidden_size, 1)


    def forward(self, x):
        feats = self.ClassifierLayer(x)
        logits = self.fc(feats)
        feats_norm = torch.norm(feats, p=2, dim=1, keepdim=True)
        pairwise_norm = torch.mm(feats_norm,
                                  torch.transpose(feats_norm, 0, 1))
        pairwise_dot = torch.mm(feats, torch.transpose(feats, 0, 1))

        # cosine similarity between feats
        pairwise_sim = pairwise_dot / pairwise_norm

        # symmetric constraint for CRF weights
        W_sym = (self.W + torch.transpose(self.W, 0, 1)) / 2
        pairwise_potential = pairwise_sim * W_sym
        unary_potential = logits.clone()

        for i in range(self.iteration):
            ## current Q after normalizing the logits
            probs = torch.transpose(logits.sigmoid(), 0, 1)

            ## taking expectation of pairwise_potential using current Q
            ## 4 cases to consider when calculating expectation: 
            ## 1. i=C,j=C; 2. i=N,j=C; 3. i=C,j=N; 4. i=N,j=N  (C=Change-point, N=Non-change-point)
            
            ## case 4
            # pairwise_potential_E = torch.sum(
            #      - (1 - probs) * pairwise_potential,
            #     dim=1, keepdim=True)
            
            ## case 1 and case 4
            # pairwise_potential_E = torch.sum(
            #     probs * pairwise_potential - (1 - probs) * pairwise_potential,
            #     dim=1, keepdim=True)

            ## case 2 and case 3
            pairwise_potential_E = torch.sum(
                - probs * pairwise_potential + (1 - probs) * pairwise_potential,
                dim=1, keepdim=True)

            logits = unary_potential + pairwise_potential_E

        return logits




dirty_acc_data = ['A5A30F76-581E-4757-97A2-957553A2C6AA']
labels_without_acc_data = ['7D9BB102-A612-4E2A-8E22-3159752F55D8', '61976C24-1C50-4355-9C49-AAE44A7D09F6', 'ECECC2AB-D32F-4F90-B74C-E12A1C69BBE2', 'FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF']


path_uuid = '/Users/seunoboru/Downloads/cv_5_folds'
path_label = '/Users/seunoboru/Desktop/NGNE/CP3106/project data/ExtraSensory/'
path_data = '/Users/seunoboru/Desktop/NGNE/CP3106/project data/watch_acc/'
result_filename = '/Users/seunoboru/Desktop/NGNE/CP3106/project data/results.h[16,48,64,128].txt'

## Get uuid
# (all uuid) 
# uuid_list = ['86A4F379-B305-473D-9D83-FC7D800180EF','81536B0A-8DBF-4D8A-AC24-9543E2E4C8E0','B9724848-C7E2-45F4-9B3F-A1F38D864495','3600D531-0C55-44A7-AE95-A7A38519464E','96A358A0-FFF2-4239-B93E-C7425B901B47','8023FE1A-D3B0-4E2C-A57A-9321B7FC755F','C48CE857-A0DD-4DDB-BEA5-3A25449B2153','61359772-D8D8-480D-B623-7C636EAD0C81','99B204C0-DD5C-4BB7-83E8-A37281B8D769','11B5EC4D-4133-4289-B475-4E737182A406','2C32C23E-E30C-498A-8DD2-0EFB9150A02E','136562B6-95B2-483D-88DC-065F28409FD2','481F4DD2-7689-43B9-A2AA-C8772227162B','797D145F-3858-4A7F-A7C2-A4EB721E133C','4E98F91F-4654-42EF-B908-A3389443F2E7','74B86067-5D4B-43CF-82CF-341B76BEA0F4','B7F9D634-263E-4A97-87F9-6FFB4DDCB36C','78A91A4E-4A51-4065-BDA7-94755F0BB3BB','0A986513-7828-4D53-AA1F-E02D6DF9561B','5119D0F8-FCA8-4184-A4EB-19421A40DE0D','9DC38D04-E82E-4F29-AB52-B476535226F2','24E40C4C-A349-4F9F-93AB-01D00FB994AF','7CE37510-56D0-4120-A1CF-0E23351428D2','59818CD2-24D7-4D32-B133-24C2FE3801E5','A7599A50-24AE-46A6-8EA6-2576F1011D81','83CF687B-7CEC-434B-9FE8-00C3D5799BE6','CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC','D7D20E2E-FC78-405D-B346-DBD3FD8FC92B','59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2','0BFC35E2-4817-4865-BFA7-764742302A2D','1538C99F-BA1E-4EFB-A949-6C7C47701B20','CA820D43-E5E2-42EF-9798-BE56F776370B','9759096F-1119-4E19-A0AD-6F16989C7E1C','BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC','1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842','A76A5AF5-5A93-4CF2-A16E-62353BB70E8A','1155FF54-63D3-4AB2-9863-8385D0BD0A13','B09E373F-8A54-44C8-895B-0039390B859F','F50235E0-DD67-4F2A-B00B-1F31ADA998B9','806289BC-AD52-4CC1-806C-0CDB14D65EB6','27E04243-B138-4F40-A164-F40B60165CF3','40E170A7-607B-4578-AF04-F021C3B0384A','4FC32141-E888-4BFF-8804-12559A491D8C','E65577C1-8D5D-4F70-AF23-B3ADB9D3DBA3','5EF64122-B513-46AE-BCF1-E62AAC285D2C','0E6184E1-90C0-48EE-B25A-F1ECB7B9714E','33A85C34-CFE4-4732-9E73-0A7AC861B27A','665514DE-49DC-421F-8DCB-145D0B2609AD','A5CDF89D-02A2-4EC1-89F8-F534FDABDD96','5152A2DF-FAF3-4BA8-9CA9-E66B32671A53','CDA3BBF7-6631-45E8-85BA-EEB416B32A3C','CCAF77F0-FABB-4F2F-9E24-D56AD0C5A82F','00EABED2-271D-49D8-B599-1D4A09240601','098A72A5-E3E5-4F54-A152-BBDA0DF7B694','BEF6C611-50DA-4971-A040-87FB979F3FC1']

# subset of the uuids
uuid_list = ['BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC', '00EABED2-271D-49D8-B599-1D4A09240601', 'CA820D43-E5E2-42EF-9798-BE56F776370B', '59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2', 'B09E373F-8A54-44C8-895B-0039390B859F', 'C48CE857-A0DD-4DDB-BEA5-3A25449B2153', 'CDA3BBF7-6631-45E8-85BA-EEB416B32A3C', 'E65577C1-8D5D-4F70-AF23-B3ADB9D3DBA3', '27E04243-B138-4F40-A164-F40B60165CF3', '61359772-D8D8-480D-B623-7C636EAD0C81']

# uuid_list = read_uuid(path_uuid)
# for i in labels_without_acc_data:
#     uuid_list.remove(i)
# for i in dirty_acc_data:
#     uuid_list.remove(i)


# relaxation for DTW
psi_list = [1]


# segment length for DTW
seg_len_list = [5]

# Hyperparameters for neural network
input_size = 3
# hidden_size_list = [32, [10,5], [32,16], [64,32]]
hidden_size_list = [32]

# training length (batch size)
# set overlap length as 1, which is equivalent to: stride_size = train_len - 1 
# train_len_list = [3,4,5,6,7,8,9,10]
train_len_list = [5]


# k-fold cross validation
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=False)


# Step1: Perform DTW
for psi in psi_list:
    for seg_len in seg_len_list:
        print('Performing DTW...')
        print('psi:%d, seg_len:%d' % (psi, seg_len))
        DTW(seg_len, path_label, path_data, uuid_list, psi)


# Step2: Train model
for psi in psi_list:
    for seg_len in seg_len_list:
        for hidden_size in hidden_size_list:
            for train_len in train_len_list:
                # Print the parameter
                print('******************************************************')
                print('psi:%d, seg_len:%d, hidden_size:%s, train_len:%d' % (psi, seg_len, str(hidden_size), train_len))
                print('******************************************************')

                # Set uuids used as data et
                dataset_uuid = uuid_list

                # Load data
                print('Loading Data...')
                x_data, y_data = load_dataset_group(dataset_uuid, path_data, path_label, seg_len, psi)
                x_data_resample, y_data_resample = under_sampling(x_data, y_data)

                # Tensorize data
                dtype = torch.FloatTensor
                # x = torch.tensor(x_data_resample).type(dtype)[0:8000]
                # y = torch.tensor(y_data_resample).type(dtype)[0:8000]
                x = torch.tensor(x_data_resample).type(dtype)
                y = torch.tensor(y_data_resample).type(dtype)
                
                stride_size = train_len - 1 # Overlap length
                j = 0 # Count for cross-validation
                result = [] # Evaluation result list
                loss_list = [] # Record loss
                auc_list = [] # Record auc

                # Training begins
                print('Traning...')
                for train_ids, test_ids in kfold.split(x):
                    j += 1
                    print('Training fold %d...' % j)
                    # Get train data and test data
                    train_x = x[train_ids]
                    train_y = y[train_ids]
                    test_x = x[test_ids]
                    test_y = y[test_ids]

                    # Batchize data
                    train_batchs_x = window_split(train_x, train_len, stride_size)
                    train_batchs_y = window_split(train_y, train_len, stride_size)
                    test_batchs_x = window_split(test_x, train_len, stride_size)
                    
                    # Construct model
                    model = cnf(input_size,hidden_size,train_len)
                    optimzer = torch.optim.SGD(model.parameters(), lr=0.001)
                    loss_func = nn.BCEWithLogitsLoss()

                    # Training
                    for epoch in range(20):
                        for i in range(len(train_batchs_x)):
                            # out,logits = model(train_batchs_x[i])
                            # feats = out.view(out.size(0), -1)
                            # output = crf(feats, logits)
                            output = model(train_batchs_x[i])
                            output = torch.squeeze(output)
                            loss = loss_func(output, train_batchs_y[i])
                            optimzer.zero_grad()
                            loss.backward()
                            optimzer.step()

                        performance = evaluate(test_batchs_x, test_y, model, stride_size)
                        AUC = performance['AUC']
                        auc_list.append(AUC)
                        loss_list.append(loss)
                    # draw_graph(loss_list,'Loss')
                    # draw_graph(auc_list,'AUC')
                    # print('===============')
                    # print(loss_list)
                    # print('===============')
                    # print(auc_list)
                    # print('===============')

                    # Evaluation for every fold
                    performance = evaluate(test_batchs_x, test_y, model, stride_size)
                    result.append(performance)

                    # if epoch%5==0:
                    #     Accuracy = performance['Accuracy']
                    #     Precision = performance['Precision']
                    #     F1 = performance['F1']
                    #     Recall = performance['Recall']
                    #     AUC = performance['AUC']
                    #     print("Accuracy:", Accuracy)
                    #     print('Precision:', Precision)
                    #     print('F1 Score:',F1)
                    #     print('Recall:',Recall)
                    #     print('AUC Sore:', AUC)
                    #     print('\n')
                    

                # Evaluation for final result
                Accuracy = 0
                Precision = 0
                F1 = 0
                Recall = 0
                AUC = 0
                for perf in result:
                    Accuracy += perf['Accuracy']
                    Precision += perf['Precision']
                    F1 += perf['F1']
                    Recall += perf['Recall']
                    AUC += perf['AUC']

                print('Final Results:')
                print("Accuracy:", Accuracy/k_folds)
                print('Precision:', Precision/k_folds)
                print('F1 Score:',F1/k_folds)
                print('Recall:',Recall/k_folds)
                print('AUC Sore:', AUC/k_folds)

                
                with open(result_filename, "a+") as file_object:
                    foo = 'psi:%d, seg_len:%d, hidden_size:%s,train_len:%d\n Accuracy:%s, Precision:%s, F1 Score:%s, Recall:%s, AUC Sore:%s\n\n' % (psi, seg_len, str(hidden_size), train_len, str(Accuracy/k_folds), str(Precision/k_folds), str(F1/k_folds), str(Recall/k_folds), str(AUC/k_folds))
                    file_object.write(foo)



