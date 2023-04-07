import json
import cherrypy
import datetime
import requests
import time
import sys

class ResourceCatalogManager(object):
    
    def __init__(self, users_file, devices_file, service_file, setting_file):
        self.usersFile=users_file
        #self.users=json.load(open(self.usersFile)) #for future usage
        self.devicesFile=devices_file
        with open(self.devicesFile, "w") as myfile:
            myfile.write("[]")
        self.devices=json.load(open(self.devicesFile))
        self.settingsFile=setting_file
        self.settings=json.load(open(self.settingsFile))
        self.service_file=service_file
        self.service_info=json.load(open(self.service_file))
        poststring="http://"+self.service_info["ip_address_service"]+":"+self.service_info["ip_port_service"]
        requests.post(poststring, json.dumps(self.settings))
        print("POSTING INFORMATION TO THE SERVICE CATALOG\n")

    exposed=True
    
    def GET(self, *uri, **parameters):  #TO HANDLE GET REQUESTS FROM APPLICATIONS AND SERVICES
        if len(uri)==1 and len(parameters)<2 :
            if uri[0]=='alldevices':
                return self.viewAllDevices()
            #elif uri[0]=='allusers':   #FOR FUTURE USE
            #    return self.viewAllUsers()
            #elif uri[0]=='user':
            #    output=self.searchUserByUserID(parameters['user_id'])
            #    if output=={}:
            #        raise cherrypy.HTTPError(400, 'user not found')
            #    return output
            elif uri[0]=='device':
                output=self.searchDevicesByDeviceID(parameters['device_id'])
                if output=={}:
                    raise cherrypy.HTTPError(400, 'device not found')
                return output
            else:
                raise cherrypy.HTTPError(400, 'incorrect URI' + {uri[0]} + ', expected alldevices, allusers, device or user')

        else:
            raise cherrypy.HTTPError(400, 'incorrect URI or PARAMETERS URI' + {len(uri)} + 'PAR {len(parameters)}')
    
    def POST(self, *uri, **parameters): #TO RECEIVE INFORMATION ON USERS AND DEVICES AND REGISTERING THEM IN THE RESOURCE
        
        if len(uri)==1 :             
            body=cherrypy.request.body.read()
            json_body=json.loads(body)
              
            if(uri[0]=='device'):
                print("DEVICE INFORMATION RECEIVED!\n")
                print("body=",json_body,"\n")
                if self.searchDevicesByDeviceID(json_body['sensor_id']) !={}:
                    raise cherrypy.HTTPError(400, 'device already present')
                #print(json_body)
                self.insertDevice(json_body)
                print("DEVICE INFORMATION REGISTERED!\n")
                self.devices=json.load(open(self.devicesFile))         
            elif(uri[0]=='user'):
                if self.searchUserByUserID(json_body['user_id']) !={}:
                    raise cherrypy.HTTPError(400, 'user already present')
                self.insertUser(json_body)
                self.devices=json.load(open(self.devicesFile))          
            else:
                raise cherrypy.HTTPError(400, 'invalid uri')  
                
        else:
            raise cherrypy.HTTPError(400, 'incorrect URI or PARAMETERS')    
        
    def PUT(self, *uri, **parameters): #TO RECEIVE INFORMATION ON USERS AND DEVICES TO BE UPDATED IN THE RESOURCE
        if len(uri)==1 :             
            body=cherrypy.request.body.read()
            json_body=json.loads(body)   
            if(uri[0]=='device'):
                print("DEVICE INFORMATION RECEIVED!\n")
                print("body=",json_body,"\n")
                if self.searchDevicesByDeviceID(json_body['sensor_id']) !={}:
                    self.updateDeviceInfo(json_body)
                    print("DEVICE INFORMATION UPDATED!\n")
                else:
                    self.insertDevice(json_body)
                    print("DEVICE INFORMATION REGISTERED!\n")
                self.devices=json.load(open(self.devicesFile))      

            elif(uri[0]=='user'):

                if self.searchUserByUserID(json_body['user_id']) !='{}':
                    self.updateUserInfo(json_body)
                else:
                    self.insertUser(json_body)
                self.devices=json.load(open(self.usersFile))        

            else:
                raise cherrypy.HTTPError(400, 'invalid uri')  
                
        else:
            raise cherrypy.HTTPError(400, 'incorrect URI or PARAMETERS')   
    
    def viewAllDevices(self): #VIEW ALL DEVICES
        return json.dumps(self.devices)        
           
    def viewAllUsers(self): #VIEW ALL USERS
        return json.dumps(self.users)
    
    def searchDevicesByDeviceID(self, ID): #SEARCH DEVICES BY DEVICE ID
        for dev in self.devices:
            if(dev['sensor_id']==ID):
                return json.dumps(dev)   
        return {}                               

    def searchUserByUserID(self, ID): #SEARCH USERS BY USER ID
        for user in self.users:
            if(user['user_id']==ID):
                return json.dumps(user)
        return {}  
                
    def updateDeviceInfo(self,device): #UPDATING THE DEVICE INFORMATION
        if len(self.devices)==0:
            return {}
        for dev in self.devices:
            if dev['ID']==device['ID']:
                dev['end-points']=device['end-points']
                dev['available_resources']=device['available_resources']
                dev['insert-timestamp']=time.time()
                break
        with open(self.devicesFile, "w") as f:
            json.dump(self.devices, f)
    
    def updateUserInfo(self,user): #UPDATING THE USER INFORMATION
        for us in self.users:
            if us['ID']==user['ID']:
                us['name']=user['name']
                us['surname']=user['surname']
                us['email']=user['email']
                break
        with open(self.usersFile, "w") as f:
            json.dump(self.users, f)
          
    def insertDevice(self, device): #INSERTING THE DEVICE
        device['insert-timestamp']=time.time()
        self.devices.append(device)
        with open(self.devicesFile, "w") as f:
            json.dump(self.devices, f) 
        return
    
    def insertUser(self, user): #INSERTING THE USER
        self.users.append(user)
        with open(self.usersFile, "w") as f:
            json.dump(self.users, f)
        return

    def removeDevices(self): #REMOVING A DEVICE
        young_devices=[]
        for dev in self.devices:
            print(dev)


            if float(dev['insert-timestamp'])-float(time.time())<1200:
                young_devices.append(dev)
        self.devices=young_devices
              
        
if __name__=="__main__":

    users_file="res_cat_users_1.json" #for future use, can be modified to be obtained through argv
    devices_file=sys.argv[2]
    setting_file=sys.argv[1]
    service_file="service_catalog_info.json"

    conf={
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True, 
        }
    }
    settings=json.load(open(setting_file))
    cherrypy.tree.mount(ResourceCatalogManager(users_file, devices_file, service_file, setting_file),'/', conf)
    cherrypy.config.update(conf)
    cherrypy.config.update({'server.socket_host':settings['ip_address']})   
    cherrypy.config.update({"server.socket_port":int(settings['ip_port'])})
    cherrypy.engine.start()
    rcm=ResourceCatalogManager(users_file, devices_file, service_file, setting_file)
    while 1:
        rcm.removeDevices()
        time.sleep(120)
    cherrypy.engine.block()    
            