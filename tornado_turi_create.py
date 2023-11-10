#!/usr/bin/python
'''Starts and runs the scikit learn server'''

# For this to run properly, MongoDB must be running
#    Navigate to where mongo db is installed and run
#    something like $./mongod --dbpath "/data/db"
#    might need to use sudo (yikes!)

# database imports
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
# when i want to run in vscode , I need switch to the env i have set and choose the right interpreter --rw

# tornado imports
import tornado.web
from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

# other imports
from pprint import PrettyPrinter

# custom imports
from basehandler import BaseHandler
import turihandlers as th
import examplehandlers as eh

# Setup information for tornado class
define("port", default=8000, help="run on the given port", type=int)

pp = PrettyPrinter(indent=4)

# Utility to be used when creating the Tornado server
# Contains the handlers and the database connection
class Application(tornado.web.Application):
    def __init__(self):
        '''Store necessary handlers,
           connect to database
        '''


        # CHOOSE HERE IS YOU WANT TURI OR SKLEARN!!!
        handlers = [(r"/[/]?", BaseHandler),
                    (r"/Handlers[/]?",        th.PrintHandlers),
                    (r"/AddDataPoint[/]?",    th.UploadLabeledDatapointHandler),
                    (r"/GetNewDatasetId[/]?", th.RequestNewDatasetId),
                    (r"/UpdateModel[/]?",     th.UpdateModelForDatasetIdTuri),     #update for turi beacue the confict FileNotFoundError: [Errno 2] No such file or directory: '../models/sklearn_model_dsid1.joblib'
                    (r"/PredictOne[/]?",      th.PredictOneFromDatasetIdTuri),  
                    (r"/GetExample[/]?",      eh.TestHandler),
                    (r"/DoPost[/]?",          eh.PostHandlerAsGetArguments),
                    (r"/PostWithJson[/]?",    eh.JSONPostHandler),
                    (r"/MSLC[/]?",            eh.MSLC),             
                    ]

        self.handlers_string = str(handlers)

        try:
            print('=================================')
            print('====ATTEMPTING MONGO CONNECT=====')
            self.client  = MongoClient(serverSelectionTimeoutMS=50) # local host, default port
            print(self.client.server_info()) # force pymongo to look for possible running servers, error if none running
            # if we get here, at least one instance of pymongo is running
            self.db = self.client.turidatabase # database with labeledinstances, models
            
        except ServerSelectionTimeoutError as inst:
            print('Could not initialize database connection, stopping execution')
            print('Are you running a valid local-hosted instance of mongodb?')
            print('   Make sure that mongo db is installed and running')
            print('   using the "brew services start mongodb-community@6.0" command')
            #raise inst
        
        self.clf = {} # change it to dictionary,  the classifier model (in-class assignment, you might need to change this line!)
        # but depending on your implementation, you may not need to change it  ¯\_(ツ)_/¯

        print('=================================')
        print('==========HANDLER INFO===========')
        pp.pprint(handlers)

        settings = {'debug':True}
        tornado.web.Application.__init__(self, handlers, **settings)

    def __exit__(self):
        self.client.close() # just in case


def main():
    '''Create server, begin IOLoop 
    '''
    tornado.options.parse_command_line()
    http_server = HTTPServer(Application(), xheaders=True)
    http_server.listen(options.port)
    IOLoop.instance().start()

if __name__ == "__main__":
    main()
