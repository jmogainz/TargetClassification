"""
Remove all files in directory that are version 3 and version 1
"""
import os
import pandas as pd

guardian_dir = "Missile_Capture-Classification-Guard_Missile"

def remove_version_1_and_3(guardian_dir):
    for file in os.listdir(os.path.join(os.getcwd(), guardian_dir)):
        if file.endswith("3.csv"):
            os.remove(os.path.join(os.getcwd(), guardian_dir, file))
        if file.endswith("1.csv"):
            os.remove(os.path.join(os.getcwd(), guardian_dir, file))

def sep_data(df, class_container):
    # separate data into dataframes based on classes specified in class_container
    class_frames = {}
    for label in class_container:
        df_class = df[df["Class"] == dict_specific_data[label][0]]
        df_class = df_class[df_class["Subclass"] == dict_specific_data[label][1]]
        df_class = df_class[df_class["Type"] == dict_specific_data[label][2]]
        df_class = df_class[df_class["Subtype"] == dict_specific_data[label][3]]
        class_frames[label] = df_class

    return class_frames

def slice_x(container):
    # if dataframe, use pandas slice; else, use numpy slice
    if isinstance(container, pd.DataFrame):
        x_num_df = container[features_combined[0:7]].copy(deep=True)
        x_sCov_df = container[features_combined[7:43]].copy(deep=True)
        x_rCov_df = container[features_combined[43:52]].copy(deep=True)
        return x_num_df, x_sCov_df, x_rCov_df
    else:
        x_num_np = container[:, 0:7]
        x_sCov_np = container[:, 7:43]
        x_rCov_np = container[:, 43:52]
        return x_num_np, x_sCov_np, x_rCov_np

def remove_outliers(df):
    # loop through only the columns that are not the class
    for col in df.columns[:-4]:
        mean = df[col].mean()
        sd = df[col].std()
        if col == "range":
            df = df[df[col] >= 300]
        else:
            df = df[(df[col] <= mean+(4*sd))] #removes top 1% of data
            df = df[(df[col] >= mean-(4*sd))] #removes bottom 1% of data

    return df

def one_hot_encode(df):
    """
    One hot encode the dataframe
    """
    df_one_hot = df.copy(deep=True)

    # join each column of each row 
    df_one_hot = df_one_hot.apply(lambda x: ''.join(x.values.astype(str)), axis=1)

    # reshape for single feature
    df_one_hot = df_one_hot.values.reshape(-1, 1)

    # one hot encode the data
    enc = OneHotEncoder(sparse=False)
    enc.fit(df_one_hot)
    df_one_hot = enc.transform(df_one_hot)
    df_one_hot = pd.DataFrame(df_one_hot, columns=enc.categories_)
    
    return df_one_hot

if __name__ == "__main__":
    remove_version_1_and_3(guardian_dir)