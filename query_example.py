import requests

# An example to test that the server works correctly.
# It takes one sample from test DataFrame, requests prediction and compares it with the right target
if __name__ == '__main__':

    features = {
        'search_id': 10655.0,
        'feature_0': 9.0,
        'feature_1': 0.0,
        'feature_2': 0.0,
        'feature_3': 1.0,
        'feature_4': 20.0,
        'feature_5': 4.0,
        'feature_6': 40.0,
        'feature_7': 0.0,
        'feature_8': 0.0,
        'feature_9': 1.0,
        'feature_10': 0.0,
        'feature_11': 0.0,
        'feature_12': 0.0,
        'feature_13': 0.0,
        'feature_14': 1.0,
        'feature_15': 0.0,
        'feature_16': 1.0,
        'feature_17': 0.0,
        'feature_18': 0.0,
        'feature_19': 1.0,
        'feature_20': 0.2857139999999999,
        'feature_21': 0.0,
        'feature_22': 0.0,
        'feature_23': 0.0,
        'feature_24': 0.155,
        'feature_25': 0.1153,
        'feature_26': 0.128275,
        'feature_27': 0.0902,
        'feature_28': 0.114625,
        'feature_29': 0.1174,
        'feature_30': 0.081275,
        'feature_31': 0.075775,
        'feature_32': 0.04855,
        'feature_33': 0.058525,
        'feature_34': 0.4316,
        'feature_35': 0.50637,
        'feature_36': 0.46291,
        'feature_37': 0.03967,
        'feature_38': 0.39069,
        'feature_39': 0.0,
        'feature_40': 0.0,
        'feature_41': 0.0,
        'feature_42': 0.021702,
        'feature_43': 0.61,
        'feature_44': 0.31,
        'feature_45': 0.97844,
        'feature_46': 0.451426,
        'feature_47': 0.0,
        'feature_48': 0.0,
        'feature_49': 0.467377,
        'feature_50': 0.2659829999999999,
        'feature_51': 0.035987,
        'feature_52': 0.0,
        'feature_53': 0.0,
        'feature_54': 0.0,
        'feature_55': 0.0,
        'feature_56': 0.0,
        'feature_57': 0.0,
        'feature_58': 0.0,
        'feature_59': 0.0,
        'feature_60': 0.0,
        'feature_61': 0.0,
        'feature_62': 0.294515,
        'feature_63': 0.0,
        'feature_64': 0.0,
        'feature_65': 0.0,
        'feature_66': 0.0,
        'feature_67': 0.180215,
        'feature_68': 0.011001,
        'feature_69': 0.0462,
        'feature_70': 0.14883,
        'feature_71': 0.196644,
        'feature_72': 0.029267,
        'feature_73': 0.0,
        'feature_74': 0.0,
        'feature_75': 0.0,
        'feature_76': 0.0367399999999999,
        'feature_77': 0.0,
        'feature_78': 0.0
        }
    resp = requests.post(
        "http://127.0.0.1:80/predict",
        json=features
    )
    print(f"Input features: {features}")
    print(f"Predicted: {resp.json()}")
    print(f"Expected: {0.0}")
    print("----")
