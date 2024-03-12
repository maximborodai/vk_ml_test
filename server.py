import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# App creation and model loading
app = FastAPI()
model = joblib.load("./model.joblib")


class InputData(BaseModel):
    """
    Input features validation for the ML model
    """
    search_id: int
    feature_0: int
    feature_1: int
    feature_2: int
    feature_3: int
    feature_4: int
    feature_5: int
    feature_6: int
    feature_7: int
    feature_8: int
    feature_9: int
    feature_10: int
    feature_11: int
    feature_12: int
    feature_13: int
    feature_14: int
    feature_15: int
    feature_16: float
    feature_17: float
    feature_18: float
    feature_19: float
    feature_20: float
    feature_21: float
    feature_22: float
    feature_23: float
    feature_24: float
    feature_25: float
    feature_26: float
    feature_27: float
    feature_28: float
    feature_29: float
    feature_30: float
    feature_31: float
    feature_32: float
    feature_33: float
    feature_34: float
    feature_35: float
    feature_36: float
    feature_37: float
    feature_38: float
    feature_39: float
    feature_40: float
    feature_41: float
    feature_42: float
    feature_43: float
    feature_44: float
    feature_45: float
    feature_46: float
    feature_47: float
    feature_48: float
    feature_49: float
    feature_50: float
    feature_51: float
    feature_52: float
    feature_53: float
    feature_54: float
    feature_55: float
    feature_56: float
    feature_57: float
    feature_58: float
    feature_59: float
    feature_60: float
    feature_61: float
    feature_62: float
    feature_63: float
    feature_64: float
    feature_65: float
    feature_66: float
    feature_67: float
    feature_68: float
    feature_69: float
    feature_70: float
    feature_71: float
    feature_32: float
    feature_73: int
    feature_74: int
    feature_75: int
    feature_76: float
    feature_77: float
    feature_78: float


@app.post('/predict')
def predict(input: InputData):
    """
    :param input: input data from the post request
    :return: predicted type
    """
    features = [[
        input.search_id,
        input.feature_0,
        input.feature_1,
        input.feature_2,
        input.feature_3,
        input.feature_4,
        input.feature_5,
        input.feature_6,
        input.feature_7,
        input.feature_8,
        input.feature_9,
        input.feature_10,
        input.feature_11,
        input.feature_12,
        input.feature_13,
        input.feature_14,
        input.feature_15,
        input.feature_16,
        input.feature_17,
        input.feature_18,
        input.feature_19,
        input.feature_20,
        input.feature_21,
        input.feature_22,
        input.feature_23,
        input.feature_24,
        input.feature_25,
        input.feature_26,
        input.feature_27,
        input.feature_28,
        input.feature_29,
        input.feature_30,
        input.feature_31,
        input.feature_32,
        input.feature_33,
        input.feature_34,
        input.feature_35,
        input.feature_36,
        input.feature_37,
        input.feature_38,
        input.feature_39,
        input.feature_40,
        input.feature_41,
        input.feature_42,
        input.feature_43,
        input.feature_44,
        input.feature_45,
        input.feature_46,
        input.feature_47,
        input.feature_48,
        input.feature_49,
        input.feature_50,
        input.feature_51,
        input.feature_52,
        input.feature_53,
        input.feature_54,
        input.feature_55,
        input.feature_56,
        input.feature_57,
        input.feature_58,
        input.feature_59,
        input.feature_60,
        input.feature_61,
        input.feature_62,
        input.feature_63,
        input.feature_64,
        input.feature_65,
        input.feature_66,
        input.feature_67,
        input.feature_68,
        input.feature_69,
        input.feature_70,
        input.feature_71,
        input.feature_32,
        input.feature_73,
        input.feature_74,
        input.feature_75,
        input.feature_76,
        input.feature_77,
        input.feature_78
    ]]
    prediction = model.predict(features).tolist()[0]
    return {
        "prediction": prediction
    }


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
