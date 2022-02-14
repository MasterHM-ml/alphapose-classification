from JSON import read_json
import numpy as np
import pandas as pd
from collections import defaultdict


class Embedding():
    def __init__(self, data: dict):
        self.image_id: str = data['image_id']
        self.category_id: int = data['category_id']
        self.keypoints = data['keypoints']
        self.score: float = data['score']
        self.box: list = data['box']
        self.idx: list = data['idx']


def frontman_snap(data: list):
    
    uniques_images_areas = defaultdict()
    uniques_images_index = defaultdict()
    for i, datarow in enumerate(data):
        area = datarow['box'][0] * datarow['box'][1]
        if datarow['image_id'] in uniques_images_areas:
            if area > uniques_images_areas[datarow['image_id']]:
                uniques_images_areas[datarow['image_id']] = area
                uniques_images_index[datarow['image_id']] = i
            else:
                continue
        else:
            print(datarow['image_id'])
            uniques_images_areas[datarow['image_id']] = area
            uniques_images_index[datarow['image_id']] = i

    filtered_data = []; indexes = uniques_images_index.values()
    for i in range(len(data)):
        if i in indexes:
            filtered_data.append(data[i])
    return filtered_data


def convert_to_numpy(data: list):
    data_array = np.ndarray([len(data), 34])
    max_value = 0; 
    for row in range(len(data)):
        col = 0
        for tracer, j in enumerate(data[row]['keypoints']):
            if (tracer+1) % 3 == 0 and tracer > 0:
                tracer+=1
                continue
            else:
                if j>max_value:
                    max_value = j
                data_array[row][col] = j
                col+=1; tracer+=1
    print('Training set max value is: ', max_value)
    return np.round(data_array / max_value, 4)


def pose_output_joints() -> list:
    columns_names = [
    "Nosex", "Nosey",
    "LEyex", "LEyey",
    "REyex", "REyey",
    "LEarx", "LEary",
    "REarx", "REary",
    "LShoulderx", "LShouldery",
    "RShoulderx", "RShouldery",
    "LElbowx", "LElbowy",
    "RElbowx", "RElbowy",
    "LWristx", "LWristy",
    "RWristx", "RWristy",
    "LHipx", "LHipy",
    "RHipx", "RHipy",
    "LKneex", "LKneey",
    "Rkneex", "Rkneey",
    "LAnklex", "LAnkley",
    "RAnklex", "RAnkley",
    ]
    return columns_names

def convert_to_pandas(data):
    """
    Reduce Keypoints by filtering out Confidence score of each joint and return x,y for each.
    Args:
        f: path to json file
    Return:
        DataFrame
    """
    # data = read_json(f) # return dict
    # frontman_data = frontman_snap(data)
    data_array = convert_to_numpy(data)
    dataframe = pd.DataFrame(data=data_array, columns=pose_output_joints())
    dataframe.to_csv('data.csv')
    return dataframe

def return_data(path_to_json: str = 'alphapose-results.json'):
    """
    Reads a json file. Convert it to pandas frame. Parse into embedding class objects.
    Args:
        path_to_json: path to json file
    Return:
        list of embedding class objects, pandas data frame
    """
    json = read_json(path_to_json)
    filtered_json = frontman_snap(json)
    df = convert_to_pandas(filtered_json)
    
    embeddings = list()
    for i, data in enumerate(filtered_json):
        refined_keypoints = df.iloc[i].to_dict()
        loaded = Embedding(data=data)
        loaded.keypoints = refined_keypoints
        embeddings.append(loaded)

    return embeddings, df

if __name__=="__main__":
    x, _ = return_data()
    print(len(x), len(x[0].keypoints))
