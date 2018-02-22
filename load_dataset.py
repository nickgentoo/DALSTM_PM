"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import pandas
from datetime import datetime
import time
import numpy as np
from keras.preprocessing.text import one_hot

def buildOHE(index,n):
    L=[0]*n
    L[index]=1
    return L


def load_dataset(name):
    if name=="HELPDESK17":
        return _load_dataset_name("data/Helpdesk2017_anonimyzed.csv")
    elif name=="BPI12":
        return _load_dataset_name("data/BPI_12_anonimyzed.csv")
    elif name == "BPI12OEA":
        return _load_dataset_name("data/BPI_12_oneEndAct_anonymized.csv")
    elif name=="BPI14":
        return _load_dataset_name("data/BPI_14_anonimyzed.csv")
def _load_dataset_name(filename):
    dataframe = pandas.read_csv(filename, header=0)
    dataframe = dataframe.replace(r's+', 'empty', regex=True)
    dataframe = dataframe.fillna(0)
    #print dataframe.dtypes
    #print dataframe.select_dtypes(['float64','int64'])



    dataset=dataframe.values
    #print dataset[0]
    #dataset=dataset[:,:8]
    values = []
    for i in range(dataset.shape[1]):
        values.append(len(np.unique(dataset[:, i])) )#+1
    #print values
    #exit(1)
    #print np.unique(dataset[:, 5])
    elems_per_fold = int(values[0] / 3)

    print "DEBUG: elemns per fold",elems_per_fold
    datasetTR = dataset[dataset[:,0]<2*elems_per_fold]
    #test set
    datasetTS = dataset[dataset[:,0]>=2*elems_per_fold]
    #trick empty column siav log
    #datasetTR=datasetTR[:,:8]
    #datasetTS=datasetTS[:,:8]

    #print len(values)
    #print dataset[0]
    def generate_set(dataset):

        data=[]
        newdataset=[]
        temptarget=[]
        #analyze first dataset line
        caseID=dataset[0][0]
        event=dataset[0][1]
        starttime=datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
        lastevtime=datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
        t=time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
        n=1
        temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S"))))
        a=[(datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))-starttime).total_seconds() ]
        a.append((datetime.fromtimestamp(time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))-lastevtime).total_seconds() )
        a.append(timesincemidnight)
        a.append(datetime.fromtimestamp(time.mktime(t)).weekday()+1)
        a.extend(buildOHE(one_hot(dataset[0][1], values[1], split="|")[0], values[1]))

        field = 3
        for i in dataset[0][3:]:
            if not np.issubdtype(dataframe.dtypes[field], np.number):
                #print field
                a.extend(buildOHE(one_hot(str(i), values[field], split="|")[0],values[field] ))
            else:
                a.append(i)
            field+=1
        newdataset.append(a)
        for line in dataset[1:,:]:
            #print line
            case=line[0]
            if case==caseID:
                #print "case", case
                #continues the current case

                t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
                temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                a=[(datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))- starttime).total_seconds()]
                a.append(( datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))- lastevtime).total_seconds() )
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday()+1)

                lastevtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                a.extend(buildOHE(one_hot(line[1], values[1], split="|")[0],values[1]) )

                field=3
                for i in line[3:]:
                    if not np.issubdtype(dataframe.dtypes[field], np.number):
                        #print "object", field
                        a.extend(buildOHE(one_hot(str(i), values[field], split="|")[0], values[field]))
                    else:
                        a.append(i)
                    field+=1
                newdataset.append(a)
                n+=1
                finishtime=datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
            else:
                caseID=case
                for i in xrange(1,len(newdataset)): # +1 not adding last case. target is 0, not interesting. era 1
                    data.append(newdataset[:i])
                    #print newdataset[:i]
                newdataset=[]
                starttime = datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
                lastevtime = datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()

                a=[(datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))- starttime).total_seconds() ]
                a.append((datetime.fromtimestamp(
                    time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))-lastevtime).total_seconds() )
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday()+1)

                a.extend(buildOHE(one_hot(line[1], values[1], split="|")[0],values[1]) )

                field=3
                for i in line[3:]:
                    if not np.issubdtype(dataframe.dtypes[field], np.number):
                        a.extend(buildOHE(one_hot(str(i), values[field], split="|")[0], values[field]))
                    else:
                        a.append(i)
                    field+=1
                newdataset.append(a)
                for i in range(n): # era n
                    temptarget[-(i+1)]=(finishtime-temptarget[-(i+1)]).total_seconds()
                temptarget.pop() #remove last element with zero target
                temptarget.append(datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                finishtime=datetime.fromtimestamp(time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))


                n = 1

        #last case
        for i in xrange(1, len(newdataset) ): #+ 1 not adding last event, target is 0 in that case. era 1
            data.append(newdataset[:i])
            #print newdataset[:i]
        for i in range(n): # era n. rimosso esempio con singolo evento
            temptarget[-(i + 1)] = (finishtime - temptarget[-(i + 1)]).total_seconds()
            #print temptarget[-(i + 1)]
        temptarget.pop()  # remove last element with zero target

        #print temptarget
        print "Generated dataset with n_samples:", len(temptarget)
        assert(len(temptarget)== len(data))
        #print temptarget
        return data, temptarget
    return generate_set(datasetTR), generate_set(datasetTS)








