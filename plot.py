import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, accuracy_score
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

WEATHER_CLASSES = [0,1,2,3,5,6]
TRACK_CLASSES = [1, 2, 3, 4, 5, 7, 11, 15, 18, 19, 26, 27, 28]
TIME_OFFSETS = [5,10,15,30,60]

def plot_class_histograms_to_row(ax, df_timed_dct_single_time, time_offset, max_class_idx=6):
    """Takes a dict for single time_offset"""

    y_train = df_timed_dct_single_time['train']["TARGET_WEATHER"].to_numpy()
    y_val = df_timed_dct_single_time['val']["TARGET_WEATHER"].to_numpy()
    y_test = df_timed_dct_single_time['test']["TARGET_WEATHER"].to_numpy()

    train_counts, train_classes = np.histogram(
        y_train, bins=np.arange(max_class_idx+1))
    val_counts, val_classes = np.histogram(
        y_val, bins=np.arange(max_class_idx+1))
    test_counts, test_classes = np.histogram(
        y_test, bins=np.arange(max_class_idx+1))

    ax.bar(np.arange(max_class_idx)-0.2, train_counts,
           0.2, edgecolor='k', label='Train')
    ax.bar(np.arange(max_class_idx), val_counts,
           0.2, edgecolor='k', label='Val')
    ax.bar(np.arange(max_class_idx)+0.2, test_counts,
           0.2, edgecolor='k', label='Test')

    ax.set_title(f'Time Offset: {time_offset}', fontsize=15)
    ax.set_xlabel('Classes', fontsize=13)
    ax.set_ylabel('Counts', fontsize=13)
    ax.set_xticks(np.arange(max_class_idx+1))
    ax.grid()
    ax.legend(loc=1, fontsize=13)


def plot_class_histograms(df_timed_dct, max_class_idx=6):
    fig, axs = plt.subplots(5, 1, figsize=(8, 16), constrained_layout=True)
    fig.suptitle('Class Histograms of Each Time Step', fontsize=20)
    for i, (time_offset, df_timed_dct_single_time) in enumerate(df_timed_dct.items()):
        plot_class_histograms_to_row(
            axs[i], df_timed_dct_single_time, time_offset, max_class_idx)
    plt.show()
    return fig


def handle_categorical(df, categorical_cols):
    for col in categorical_cols:
        for c in categorical_cols[col]:
            df[f"{col}_{c}"] = \
            df[f"{col}"].apply(lambda row: 1 if row == c else 0)
    df = df.drop(categorical_cols, axis=1)
    return df


def preprocess(Xs, categorical_cols):
    res_Xs = []
    for X in Xs:
        X = handle_categorical(X, categorical_cols)    
        res_Xs.append(X)
    return res_Xs


def save_model(model, model_type, time_offset):
    os.makedirs(os.path.join("model_weights", model_type), exist_ok=True)
    file_path = f'model_weights/{model_type}/{time_offset}.pkl'
    pickle.dump(model, open(file_path, 'wb'))

    
def add_weather_preds(Xs, time_offset):
    model = pickle.load(open(f'model_weights/weather/{time_offset}.pkl', 'rb'))
    res_Xs = []
    cat_cols = []
    for X in Xs:
        cols= {}
        for to in TIME_OFFSETS:
            if to < int(time_offset):
                cols[f"WHEATHER_PRED_{to}"] = model.predict(X)
                cat_cols.append(f"WHEATHER_PRED_{to}")
        add_df = pd.DataFrame.from_dict(cols)
        X = pd.concat([X, add_df], axis=1)
        res_Xs.append(X)
    cat_cols = list(set(cat_cols))
    cat_cols_dct = {c: WEATHER_CLASSES for c in cat_cols}
    res_Xs = preprocess(res_Xs, cat_cols_dct)
    return res_Xs


def get_predict(dfs, model, metric_func, model_type, time_offset, target_col, drop_cols, 
                categorical_cols):
    X_train = dfs["train"].drop(drop_cols, axis=1)
    y_train = dfs["train"][target_col]
    X_val = dfs["val"].drop(drop_cols, axis=1)
    y_val = dfs["val"][target_col]
    X_test = dfs["test"].drop(drop_cols, axis=1)
    y_test = dfs["test"][target_col]
    
    
    
    X_train,X_val, X_test = preprocess([X_train,X_val, X_test], categorical_cols)
    
    # if model_type == "rain_percentage":
    #     X_train, X_val, X_test = add_weather_preds([X_train,X_val, X_test], time_offset)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    
    
    
    model.fit(X_train, y_train)
    save_model(model, model_type, time_offset)
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    return metric_func(y_train, y_val, y_test, y_pred_train, y_pred_val, y_pred_test)


# def get_predict(dfs, model, problem_name, target_col, drop_cols, metric_func):
#     X_train = dfs["train"].drop(drop_cols, axis=1)  # remove annoying warnings
#     y_train = dfs["train"][target_col].to_numpy().reshape(-1)
#     X_val = dfs["val"].drop(drop_cols, axis=1)
#     y_val = dfs["val"][target_col].to_numpy().reshape(-1)
#     X_test = dfs["test"].drop(drop_cols, axis=1)
#     y_test = dfs["test"][target_col].to_numpy().reshape(-1)
#     model.fit(X_train, y_train)
#     y_pred_train = model.predict(X_train)
#     y_pred_val = model.predict(X_val)
#     y_pred_test = model.predict(X_test)
#     #print(problem_name)
#     return metric_func(y_train, y_val, y_test, y_pred_train, y_pred_val, y_pred_test)


def get_classification_metrics(y_train, y_val, y_test, y_pred_train, y_pred_val, y_pred_test, verbose=False):
    y_train = y_train["TARGET_WEATHER"]
    y_test = y_test["TARGET_WEATHER"]
    y_val = y_val["TARGET_WEATHER"]
    
    # Find exactly which classes are in each set
    train_classes = np.unique(np.concatenate(
        (y_train, y_pred_train))).astype('int32')
    val_classes = np.unique(np.concatenate(
        (y_val, y_pred_val))).astype('int32')
    test_classes = np.unique(np.concatenate(
        (y_test, y_pred_test))).astype('int32')
    # Deal with class imbalances
    max_class_id = np.max(
        [np.max(train_classes), np.max(val_classes), np.max(test_classes)])

    # Calculate scores
    train_acc = accuracy_score(y_train, y_pred_train)
    train_prec, train_rec, train_f1, train_sup = precision_recall_fscore_support(
        y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_prec, val_rec, val_f1, val_sup = precision_recall_fscore_support(
        y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec, test_rec, test_f1, test_sup = precision_recall_fscore_support(
        y_test, y_pred_test)

    train_prec_full = np.zeros((max_class_id+1,))
    train_rec_full = np.zeros((max_class_id+1,))
    train_f1_full = np.zeros((max_class_id+1,))
    train_acc_full = np.zeros((max_class_id+1,))
    train_prec_full[train_classes] = train_prec
    train_rec_full[train_classes] = train_rec
    train_f1_full[train_classes] = train_f1
    train_acc_full[train_classes] = train_acc

    val_prec_full = np.zeros((max_class_id+1,))
    val_rec_full = np.zeros((max_class_id+1,))
    val_f1_full = np.zeros((max_class_id+1,))
    val_acc_full = np.zeros((max_class_id+1,))
    val_prec_full[val_classes] = val_prec
    val_rec_full[val_classes] = val_rec
    val_f1_full[val_classes] = val_f1
    val_acc_full[val_classes] = val_acc

    test_prec_full = np.zeros((max_class_id+1,))
    test_rec_full = np.zeros((max_class_id+1,))
    test_f1_full = np.zeros((max_class_id+1,))
    test_acc_full = np.zeros((max_class_id+1,))
    test_prec_full[test_classes] = test_prec
    test_rec_full[test_classes] = test_rec
    test_f1_full[test_classes] = test_f1
    test_acc_full[test_classes] = test_acc


    prec = np.stack([train_prec_full, val_prec_full, test_prec_full])
    rec = np.stack([train_rec_full, val_rec_full, test_rec_full])
    f1 = np.stack([train_f1_full, val_f1_full, test_f1_full])
    acc = np.stack([train_acc_full, val_acc_full, test_acc_full])

    #classes = [train_classes,val_classes,test_classes]

    return prec, rec, f1, acc


def get_regression_metrics(y_train, y_val, y_test, y_pred_train, y_pred_val, y_pred_test):
    print("Train MAE: {}".format(mean_absolute_error(y_train, y_pred_train)))
    print("Val MAE: {}".format(mean_absolute_error(y_val, y_pred_val)))
    print("Test MAE: {}".format(mean_absolute_error(y_test, y_pred_test)))
    print()


def plot_model_performance(model, data, model_type, save=False):
    if model_type == "weather":    
        fig, ax = plt.subplots(5, 4, figsize=(15, 16), constrained_layout=True)
        title = "{} Performances".format(type(model).__name__)
        fig.suptitle(title, fontsize=20)
        for i, (time_offset, data_time) in enumerate(data.items()):
            precisions, recalls, f1_scores, accs = \
                get_predict(data_time, model,get_classification_metrics, "weather", time_offset,
                    target_col=["TARGET_WEATHER"], 
                    drop_cols=["TARGET_WEATHER","TARGET_RAIN_PERCENTAGE"], 
                    categorical_cols={"M_WEATHER_FORECAST_SAMPLES_M_WEATHER" : WEATHER_CLASSES})
                # get_predict(data_time,
                #                                          model,
                #                                          f"Weather {time_offset} Min",
                #                                          ["TARGET_WEATHER"],
                #                                          ["TARGET_WEATHER",
                #                                              "TARGET_RAIN_PERCENTAGE"],
                #                                          get_classification_metrics)
                
                                                        

            plot_scores_to_ax(ax[i], precisions, recalls, f1_scores, accs, time_offset)
        plt.show()
        if save:
            fig.savefig('{}.jpeg'.format('_'.join(title.split(' '))))
    else:
        for i, (time_offset, data_time) in enumerate(data.items()):
            get_predict(data_time, model, get_regression_metrics, "rain_percentage", time_offset,
                        target_col=["TARGET_RAIN_PERCENTAGE"], 
                        drop_cols=["TARGET_WEATHER","TARGET_RAIN_PERCENTAGE"], 
                        categorical_cols={"M_WEATHER_FORECAST_SAMPLES_M_WEATHER" : WEATHER_CLASSES})

def plot_scores_to_ax(ax, precisions, recalls, f1_scores, accs, time_offset):

    ax[0].set_title('{} Min\nPrecision Scores'.format(
        time_offset), fontsize=15)
    ax[0].bar(np.arange(len(precisions[0]))-0.2,
              precisions[0], 0.2, edgecolor='k', label='Train')
    ax[0].bar(np.arange(len(precisions[1])), precisions[1],
              0.2, edgecolor='k', label='Val')
    ax[0].bar(np.arange(len(precisions[2]))+0.2,
              precisions[2], 0.2, edgecolor='k', label='Test')
    max_no_prec = max([len(precisions[0]), len(
        precisions[1]), len(precisions[2])])
    ax[0].set_xticks(np.arange(max_no_prec))
    #ax[0].set_xticks(classes[0])

    ax[1].set_title('{} Min\nf1 Scores'.format(time_offset), fontsize=15)
    ax[1].bar(np.arange(len(f1_scores[0]))-0.2, f1_scores[0],
              0.2, edgecolor='k', label='Train')
    ax[1].bar(np.arange(len(f1_scores[1])), f1_scores[1],
              0.2, edgecolor='k', label='Val')
    ax[1].bar(np.arange(len(f1_scores[2]))+0.2,
              f1_scores[2], 0.2, edgecolor='k', label='Test')
    max_no_f1 = max([len(f1_scores[0]), len(f1_scores[1]), len(f1_scores[2])])
    ax[1].set_xticks(np.arange(max_no_f1))
    #ax[1].set_xticks(classes[1])

    ax[2].set_title('{} Min\nRecall Scores'.format(time_offset), fontsize=15)
    ax[2].bar(np.arange(len(recalls[0]))-0.2, recalls[0],
              0.2, edgecolor='k', label='Train')
    ax[2].bar(np.arange(len(recalls[1])), recalls[1],
              0.2, edgecolor='k', label='Val')
    ax[2].bar(np.arange(len(recalls[2]))+0.2, recalls[2],
              0.2, edgecolor='k', label='Test')
    max_no_recalls = max([len(recalls[0]), len(recalls[1]), len(recalls[2])])
    ax[2].set_xticks(np.arange(max_no_recalls))
    #ax[2].set_xticks(classes[2])

    ax[3].set_title('{} Min\nAccuracy Scores'.format(time_offset), fontsize=15)
    ax[3].bar(np.arange(len(accs[0]))-0.2, accs[0],
              0.2, edgecolor='k', label='Train')
    ax[3].bar(np.arange(len(accs[1])), accs[1],
              0.2, edgecolor='k', label='Val')
    ax[3].bar(np.arange(len(accs[2]))+0.2, accs[2],
              0.2, edgecolor='k', label='Test')
    max_no_accs = max([len(accs[0]), len(accs[1]), len(accs[2])])
    ax[3].set_xticks(np.arange(max_no_accs))

    for i, x in enumerate(ax.flatten()):
        x.grid()
        x.legend(loc="lower right")
        x.set_yticks(np.arange(0, 1.1, 0.1))
        x.set_xlabel('Classes')
        x.set_ylabel('Scores')
        #x.set_xlim([-0.4,classes[i].max()+0.4])
    return
