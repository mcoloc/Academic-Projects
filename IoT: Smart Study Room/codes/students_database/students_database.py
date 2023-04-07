import json
import cherrypy
import requests
from RPLCD.gpio import CharLCD
from RPi import GPIO
import sys

#we build it just to simulate the correctness of the usage of the sensor evo, but theoretically it is external to our system

class StudentsDatabaseManager(object): 
    
    def __init__(self):
        self.settings="evo_server_config.json"
        self.conf=json.load(open(self.settings))        
    exposed=True
    
    def GET(self, *uri, **parameters): #HANDLING GET REQUEST FROM SENSOR EVO, PROVIDING THE JSON WITH THE BOOKED STUDENTS
        self.booked_students_file="booked_students.json"
        self.booked_students=json.load(open(self.booked_students_file)) 
        if len(uri)==1:
            if uri[0]=='all_bookings':
                return json.dumps(self.booked_students)
        else:
                error_string="incorrect URI or PARAMETERS URI"+ {len(uri)} +"PAR"+ {len(parameters)}
                raise cherrypy.HTTPError(400, error_string)

    def POST(self): #RECEIVING ACCESS INFORMATION FROM SENSOR EVO AND PRINTING IT ON LCD SCREEN
        body=cherrypy.request.body.read()
        GPIO.setwarnings(False)
        lcd = CharLCD(numbering_mode=GPIO.BOARD, cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[40, 38, 36, 32, 33, 31, 29, 23])
        lcd.write_string(body)
    
    def getPort(self):
        return self.conf['port']   
    
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
        registration(setting_file, service_file)
    else:
        rc=rc['result']
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

    registration(sys.argv[1], "service_catalog_info.json")
    config=json.load(open("evo_server_config.json"))

    conf={
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True, 
        }
    }
    cherrypy.tree.mount(StudentsDatabaseManager(),'/', conf)
    cherrypy.config.update(conf)
    cherrypy.config.update({"server.socket_host":config["ip"]})
    cherrypy.config.update({"server.socket_port":config["port"]})
    cherrypy.engine.start()            