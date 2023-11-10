#!/usr/bin/python

import os
from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

import turicreate as tc
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

import pickle
from bson.binary import Binary
import json
import numpy as np

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        label = data['label']
        sess  = data['dsid']

        dbid = self.db.labeledinstances.insert_one(
            {"feature":fvals,"label":label,"dsid":sess}
            );
        self.write_json({"id":str(dbid),
            "feature":[str(len(fvals))+" Points Received",
                    "min of: " +str(min(fvals)),
                    "max of: " +str(max(fvals))],
            "label":label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModelForDatasetIdTuri(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = self.get_int_arg("dsid",default=0)

        data = self.get_features_and_labels_as_SFrame(dsid)

        # fit the model to the data
        acc = -1
        best_model = 'unknown'
        if len(data)>0:
            
            model = tc.classifier.create(data,target='target',verbose=0)# training
            yhat = model.predict(data)
            self.clf[dsid] = model  #change the clf to dictionary and set the model to the dsid
            print(self.clf)
            acc = sum(yhat==data['target'])/float(len(data))
            # save model for use later, if desired
            model.save('../models/turi_model_dsid%d'%(dsid))
            

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})

    def get_features_and_labels_as_SFrame(self, dsid):
        # create feature vectors from database
        features=[]
        labels=[]
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            features.append([float(val) for val in a['feature']])
            labels.append(a['label'])

        # convert to dictionary for tc
        data = {'target':labels, 'sequence':np.array(features)}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

class PredictOneFromDatasetIdTuri(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    
        fvals = self.get_features_as_SFrame(data['feature'])
        dsid  = data['dsid']

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!
        # Part three
        # get the model for the clf dictionary
        # Accessing the model for a specific dsid
        if dsid in self.clf:
            model = self.clf[dsid]
            print(f"Model for dsid {dsid} not found in dic.")
            predLabel= model.predict(fvals);
            self.write_json({"prediction":str(predLabel)})
        # Now you can use turi_model as needed
        else:
        # Handle the case where dsid is not in the dictionary
            # before i load the model file i should check is it exist 
            model_path = f'../models/turi_model_dsid{dsid}'
            if os.path.exists(model_path):
              # The directory exists, load the model
                model = tc.load_model(model_path)
                model=tc.load_model('../models/turi_model_dsid%d'%(dsid))
                self.clf[dsid]=model #after load add to the dictionary
                predLabel= model.predict(fvals);
                self.write_json({"prediction":str(predLabel)})
            else:
              # The directory doesn't exist, handle the error accordingly
              # Means trained not done 
                print(f"Error: Directory {model_path} does not exist.")
                self.write_json({"error":"Wait for the model trainning"})

            
            

        # original method
        # if(self.clf == []):
        #     print('Loading Model From file')
        #     self.clf = tc.load_model('../models/turi_model_dsid%d'%(dsid))

        # predLabel = self.clf.predict(fvals);
        # self.write_json({"prediction":str(predLabel)})

    def get_features_as_SFrame(self, vals):
        # create feature vectors from array input
        # convert to dictionary of arrays for tc

        tmp = [float(val) for val in vals]
        tmp = np.array(tmp)
        tmp = tmp.reshape((1,-1))
        data = {'sequence':tmp}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

# TODO: test out the sklearn dataset responding 
class UpdateModelForDatasetIdSklearn(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = self.get_int_arg("dsid",default=0)

        # create feature vectors and labels from database
        features = []
        labels   = []
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            features.append([float(val) for val in a['feature']])
            labels.append(a['label'])

        # fit the model to the data
        model = KNeighborsClassifier(n_neighbors=1);
        acc = -1;
        if labels:
            model.fit(features,labels) # training
            lstar = model.predict(features)
            self.clf[dsid]= model    # change the clf to dictionary to store the model and the key is dsid
            acc = sum(lstar==labels)/float(len(labels))
            print(self.clf)
            # just write this to model files directory
            dump(model, '../models/sklearn_model_dsid%d.joblib'%(dsid))


        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})




class PredictOneFromDatasetIdSklearn(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    

        vals = data['feature'];
        fvals = [float(val) for val in vals];
        fvals = np.array(fvals).reshape(1, -1)
        dsid  = data['dsid']

        # load the model (using pickle)
        if(self.clf == []):
            # load from file if needed
            print('Loading Model From DB')
            tmp = load('../models/sklearn_model_dsid%d.joblib'%(dsid)) 
            self.clf = pickle.loads(tmp['model'])

        predLabel = self.clf.predict(fvals);
        self.write_json({"prediction":str(predLabel)})



