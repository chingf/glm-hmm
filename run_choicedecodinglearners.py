# Import statements
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from LearningSession import *
from LearningChoicePredictor import *

def main(mode=None, reduce_dim=None):
    mouse = "mSM63"
    days = os.listdir("/home/chingf/engram/data/musall/learning/neural/mSM63")
    days = [d for d in days if "2018" in d]
    days.sort(key = lambda date: datetime.strptime(date, '%d-%b-%Y'))
    results = []
    for day in days:
        result = _run_decoding(mouse, day, mode, reduce_dim)
        results.append(result)
    if mode == "LOO":
        pickle.dump(results,
            open("pickles/choicedecodingloo_learning_mSM63.p", "wb")
            )
    elif mode == "LOI":
        pickle.dump(results,
            open("pickles/choicedecodingloireg_learning_mSM63.p", "wb")
            )
    else:
        pickle.dump(results,
            open("pickles/choicedecodingregreduce" +\
                    str(reduce_dim) + "_learning_mSM63.p", "wb")
            )

def _run_decoding(mouse, day, mode, reduce_dim):
    session = LearningSession(mouse, day, access_engram=True)
    predictor = LRChoice(session, reduce_dim=reduce_dim, mode=mode)
    result = predictor.fit_all()
    return result

for rd in [1.0]:
    main(reduce_dim=1.0)
