import re
import numpy as np
import pandas as pd

def clean_up_nans(df, merge_column_name, prefixes_arr):
    """ Mettre à zéro les NaN des colonnes commençant par 'prefixes_arr'.
    De plus, exploiter la colonne 'merge_column_name' pour mettre 1 si le LEFT JOIN a bien fonctionné, et 0 s'il a échoué.
    Modifie le dataframe 'df' par effet de bord """
    merge_values_set = set(df[merge_column_name].unique())
    assert len(merge_values_set) == 2
    assert 'both' in merge_values_set
    assert 'left_only' in merge_values_set

    df[merge_column_name].replace({'both':1,'left_only':0}, inplace=True)

    for prefix in prefixes_arr:
        cols_for_nan_replacement = [ col for col in df.columns if col.startswith(prefix)  ]
        df.loc[:,cols_for_nan_replacement] = df[cols_for_nan_replacement].fillna(value=0)

    # En BONUS : replacer les ' ' (espaces vides) dans les noms des colonnes par '_' (underscore)
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x))


def take_care_of_nan(df):
    """ Nettoyer le dataframe 'df' en retirant les NaN
    Le df est mis à jour par effet de bord """
    # Nettoyage des NaN
    clean_up_nans(df, "merge_with_bureau", ["BURO_","ACTIVE_","CLOSED_"])
    clean_up_nans(df, "merge_with_prev", ["PREV_","APPROVED_","REFUSED_"])
    clean_up_nans(df, "merge_with_pos", ["POS_"])
    clean_up_nans(df, "merge_with_ins", ["INSTAL_"])
    clean_up_nans(df, "merge_with_cc", ["CC_"])


    # Remplacer les NaN restants par 0, et se débarasser des 3 colonnes contenant inf
    df.loc[ : , df.columns != "TARGET" ] = df.loc[ : , df.columns != "TARGET" ].fillna( value=0 )
    
    
    cols_withs_infs_series = pd.DataFrame(np.isinf(df.values), columns=df.columns).sum()
    cols_to_drop = list(cols_withs_infs_series.loc[ cols_withs_infs_series > 0 ].index)
    df.drop(labels=cols_to_drop,axis="columns",inplace=True)


def work_exam_split(df):
    """ Séparation du DataFrame en deux sous-DF (un étiqueté, l'autre pas étiqueté) : ce sont le "jeu de travail" et le "jeu d'examination"
    On se base sur l'index, qui retombe à 0 pour le jeu d'exam """
    train_test_boundaries_list = list(df.query( "index == 0 ").index)
    assert len( train_test_boundaries_list) == 2

    loans_work_df = df.iloc[0:train_test_boundaries_list[1]]
    loans_exam_df = df.iloc[train_test_boundaries_list[1]:]    

    return (loans_work_df, loans_exam_df)