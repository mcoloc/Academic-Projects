import json
import random
import requests
import RPi.GPIO as rpi
from gpiozero import LED
import sys
import cherrypy
import time

class BrightnessControlServer:
    exposed=True
    def __init__(self, pin):
        
        self.old_result=0
        self.pinL=pin
        rpi.setmode(rpi.BCM)
        rpi.setup(self.pinL, rpi.OUT)

    def POST(self): #RECEIVING MESSAGE FROM SENSOR BRIGHTNESS IN ORDER TO SWITCH ON/OFF THE LED       
        result=json.loads(cherrypy.request.body.read())
        print(result)
        if result['result']==1 and self.old_result==0:
            rpi.output(self.pinL,rpi.HIGH)
            time.sleep(3)
        if result['result']==0 and self.old_result==1:
            rpi.output(self.pinL,rpi.LOW)
            time.sleep(3)
        self.old_result=result['result']

def registration(setting_file, service_file): #IN ORDER TO REGISTER ON THE RESOURCE CATALOG
    with open(setting_file,"r") as f1:    
        conf=json.loads(f1.read())

    with open(service_file,"r") as f2:    
        conf_service=json.loads(f2.read())
    requeststring='http://'+conf_service['ip_address_service']+':'+conf_service['ip_port_service']+'/rooms_name_owner'
    r=requests.get(requeststring)
    print("INFORMATION FROM SERVICE CATALOG RECEIVED!\n")
    print(r.text)
    print("Available rooms and owners:\n "+r.text+"\n")
    owner=input("Which owner? ")
    room=input("\nWhich room? ")

    requeststring='http://'+conf_service['ip_address_service']+':'+conf_service['ip_port_service']+'/room_info?room='+room+'&owner='+owner
    r=requests.get(requeststring)
    print("INFORMATION OF RESOURCE CATALOG (room) RECEIVED!\n") #PRINT FOR DEMO
    r_c=json.loads(r.text)
    if r_c['isFound']==0:
        return 'NOT FOUND'
    else:
        rc=r_c['result']
    rc_ip=rc["ip_address"]
    rc_port=rc["ip_port"]
    poststring='http://'+rc_ip+':'+rc_port+'/device'
    rc_basetopic=rc["base_topic"]
    rc_broker=rc["broker"]
    rc_port=rc["broker_port"]
    rc_owner=rc["owner"]
     
    sensor_model=conf["sensor_model"]
    
    requeststring='http://'+conf_service['ip_address_service']+':'+conf_service['ip_port_service']+'/base_topic'
    sbt=requests.get(requeststring)

    service_b_t=json.loads(sbt.text)
    topic=[]
    body=[]
    index=0
    for i in conf["sensor_type"]:
        print(i)
        topic.append(service_b_t+'/'+rc_owner+'/'+rc_basetopic+"/"+i+"/"+conf["sensor_id"])
        body_dic= {
        "sensor_id":conf['sensor_id'],
        "sensor_type":conf['sensor_type'],
        "owner":rc["owner"],
            "measure":conf["measure"][index],
            "end-points":{
                "basetopic":service_b_t+'/'+rc_owner+'/'+rc_basetopic,
                "complete_topic":topic,
                "broker":rc["broker"],
                "port":rc["broker_port"]
            }
        }
        body.append(body_dic)
        requests.post(poststring,json.dumps(body[index]))
        print("REGISTRATION TO RESOURCE CATALOG (room) DONE!\n") #PRINT FOR DEMO
        
        index=index+1


if __name__=="__main__":

    config=json.load(open(sys.argv[1]))
    result=registration(sys.argv[1], "service_catalog_info.json")
    while result=='NOT FOUND':
        result=registration(sys.argv[1], "service_catalog_info.json")

    conf={
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True, 
        }
    }
    cherrypy.tree.mount(BrightnessControlServer(config["pin"]),'/', conf)
    cherrypy.config.update(conf)
    cherrypy.config.update({"server.socket_host":config["ip"]})
    cherrypy.config.update({"server.socket_port":config["port"]})
    cherrypy.engine.start()

        