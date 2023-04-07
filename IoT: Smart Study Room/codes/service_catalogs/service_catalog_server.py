import json
import cherrypy
import datetime
import requests

class ServiceCatalogManager(object):
    
    def __init__(self):
        self.settings="service_settings.json"
        self.conf=json.load(open(self.settings))
        self.conf["resource_catalogs"]=[]
        with open(self.settings, "w") as myfile:
            myfile.write(json.dumps(self.conf))
        
    exposed=True
    
    def GET(self, *uri, **parameters): #METHOD FOR HANDLING GET REQUESTS REGARDING RESOURCE CATALOGS (rooms), TOPICS AND BROKER
        if len(uri)==1:
            self.settings="service_settings.json"
            self.conf=json.load(open(self.settings))
            if uri[0]=='res_cat':
                return json.dumps(self.conf["resource_catalogs"])   
            elif uri[0]=='one_res_cat': 
                results=self.conf['resource_catalogs'][len(self.conf["resource_catalogs"])-1] 
                return json.dumps(results) 

            elif uri[0]=='rooms_name_owner':
                output={ "names":[]
                            }
                for entry in self.conf["resource_catalogs"]:
                    output["names"].append(entry["base_topic"]+' '+entry["owner"]+'\n')
                return json.dumps(output)

            elif uri[0]=='room_info':
                output={'isFound':0,
                        'result':''
                        }
                for entry in self.conf["resource_catalogs"]:
                    if parameters['room']==entry["base_topic"] and parameters["owner"]==entry["owner"]:
                        output['isFound']=1
                        output['result']=entry
                        return json.dumps(output)
                return json.dumps(output)

            elif uri[0]=='broker':
                output_site=self.conf['broker']
                output_port=self.conf['broker_port']
                output={                
                    'broker_port': output_port,
                    'broker': output_site, 
                }    
                print(output)
                return output

            elif uri[0]=='base_topic':
                return json.dumps(self.conf["base_topic"])
        else:
                error_string="incorrect URI or PARAMETERS URI"+ {len(uri)} +"PAR"+ {len(parameters)}
                raise cherrypy.HTTPError(400, error_string)
    
    def POST(self, *uri, **parameters):   #RECEIVING INFORMATION OF RESOURCE CATALOGS AND INSERTING THEM IN THE JSON
        
        self.settings="service_settings.json"
        self.conf=json.load(open(self.settings))
        if len(uri)==0 :             
            body=cherrypy.request.body.read()
            print("POST RECEIVED WITH BODY:", body)
            output={'isFound':0}
            json_body=json.loads(body)            
            if self.insertResCat(json_body) == "found":
                return json.dumps(output)
            else:
                output['isFound']=1
                return json.dumps(output)

        else:
            error_string="incorrect URI or PARAMETERS" + {len(uri)}
            raise cherrypy.HTTPError(400, error_string)

    def PUT(self, *uri, **parameters):   #RECEIVING INFORMATION OF RESOURCE CATALOGS AND UPDATING THEM IN THE JSON
        
        self.settings="service_settings.json"
        self.conf=json.load(open(self.settings))
        if len(uri)==0 :             
            body=cherrypy.request.body.read()
            print("\nPUT RECEIVED WITH BODY:", body)      
            json_body=json.loads(body)            
            self.insertModifiedResCat(json_body)
        else:
            error_string="incorrect URI or PARAMETERS" + {len(uri)}
            raise cherrypy.HTTPError(400, error_string)   
    
    def insertResCat(self, json_body): #METHOD FOR INSERTING RESOURCE CATALOGS 
        for entry in self.conf["resource_catalogs"]:
            if entry["base_topic"]==json_body["base_topic"] and entry["owner"]==json_body["owner"]:
                return "found"
                
        self.conf['resource_catalogs'].append(json_body)
        print(json_body)
        with open(self.settings, "w") as f:
            json.dump(self.conf, f)
        return "" 

    def insertModifiedResCat(self, json_body): #METHOD FOR UPDATING EXISTING RESOURCE CATALOGS
        for entry in self.conf["resource_catalogs"]:
            if entry["base_topic"]==json_body["base_topic"] and entry["owner"]==json_body["owner"]:
                entry['base_topic']==json_body["base_topic"]
                entry["broker"]==json_body["broker"]
                entry["broker_port"]==json_body["broker_port"]
                entry["base_topic"]==json_body["base_topic"]
                entry["ip_address"]==json_body["ip_address"]
                entry["ip_port"]==json_body["ip_port"]
                print(json_body)
        with open(self.settings, "w") as f:
            json.dump(self.conf, f)
            
    def getPort(self):
        return self.conf['ip_port']


    def getBrokerPort(self):
        return self.conf['broker_port']
    
if __name__=="__main__":
    service_info=json.load(open("service_catalog_info.json"))
    conf={
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True, 
        }
    }
    cherrypy.tree.mount(ServiceCatalogManager(),'/', conf)
    cherrypy.config.update(conf)
    cherrypy.config.update({'server.socket_host':service_info['ip_address_service']})   
    cherrypy.config.update({"server.socket_port":ServiceCatalogManager().getPort()})
    cherrypy.engine.start()
    cherrypy.engine.block()     
            