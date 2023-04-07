import telepot
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
import json
import requests
import time
import sys

        
class EchoBot1:
    def __init__(self, token, service_catalog_info,limitfile):
        
        self.tokenBot = token
        self.service_catalog_info = json.load(open(service_catalog_info))

        self.bot = telepot.Bot(self.tokenBot)
        MessageLoop(self.bot, {'chat': self.on_chat_message,
                               'callback_query': self.on_callback_query}).run_as_thread()
        poststring="http://"+self.service_catalog_info["ip_address_service"]
        
        service_get_string="http://"+self.service_catalog_info["ip_address_service"]+":"+self.service_catalog_info["ip_port_service"]+"/res_cat"
        rooms_all=json.loads(requests.get(service_get_string).text)
        self.rooms=[]
        self.rooms_warning=[] 
        self.check=0

        for entry in rooms_all:
            request_string="http://"+entry["ip_address"]+":"+entry["ip_port"]+"/alldevices"
            devices=json.loads(requests.get(request_string).text)
            sensors=[]
            for dev in devices:
                for type in dev["sensor_type"]:
                    if type != 'fiscal_code':
                        sensors.append(type)
            room={"room_name":entry["base_topic"],
                "room_sensors":sensors
                }
            found=0
            for own in self.rooms:
                if own['owner']==entry['owner']:
                    own['rooms'].append(room)
                    found=1
            if found==0:
                self.rooms.append({'owner':entry['owner'],
                      'rooms':[room]
                    })
        self.chosen_owner=0
        self.requested_owner=''
        self.chosen_room=0
        self.requested_room=''
        self.r=''
        self.limits = json.load(open(limitfile))
        
    def on_chat_message(self, msg):
        content_type, chat_type, chat_ID = telepot.glance(msg)
        message = msg['text']
        
        if message == "/start":
            self.bot.sendMessage(chat_ID, text="  Command :/operation")
        
        
        if message == "/operation":
            self.chosen_owner=0
            self.chosen_room=0
            buttons=[]
            for room in self.rooms:
                buttons.append ([InlineKeyboardButton(text=room["owner"],callback_data=room["owner"])])
            keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
            self.bot.sendMessage(chat_ID, text='Which owner are you interested in?', reply_markup=keyboard)    
        elif  message == "/check":
            self.check=1
            while 1:
                time.sleep(10)
                
                for room in self.rooms_warning:

                    for dev in room["room_sensors"]:
                        if dev!='fiscal_code':
                            print("http://"+sys.argv[1]+"/?owner="+room["owner"]+"&room_name="+room["room_name"]+"&sensor_type="+dev+"&check=value")
                            value=int(round(float(requests.get("http://"+sys.argv[1]+"/?owner="+room["owner"]+"&room_name="+room["room_name"]+"&sensor_type="+dev+"&check=value").text),0))
                            #GET REQUEST TO THE SENSOR SUBSCRIBER IN ORDER TO RECEIVE SENSOR DATA

                            for l in self.limits:
                                if dev==l["sensor_type"]:
                                    if value<l["min"]:
                                        self.bot.sendMessage(chat_ID, text="WARNING! "+ dev+" is low in "+room["room_name"]+' - '+room["owner"] )
                                    elif value>l["max"]:
                                        self.bot.sendMessage(chat_ID, text="WARNING! "+ dev+" is very high in "+room["room_name"]+' - '+room["owner"]  )
                                    elif value>l["max_good"]:
                                        self.bot.sendMessage(chat_ID, text="WARNING! "+ dev+" is  high in "+room["room_name"]+' - '+room["owner"]  )  

        else:
            self.bot.sendMessage(chat_ID, text="Command not supported")
        

    def on_callback_query(self,msg):
        query_ID , chat_ID , query_data = telepot.glance(msg,flavor='callback_query')
        message = query_data
        
        for owner in self.rooms:
                if owner["owner"]==self.requested_owner:
                    for room in owner["rooms"]:
                        if message == room["room_name"]:
                            checked_room={"room_name":room["room_name"], "owner":owner["owner"], "room_sensors":room["room_sensors"]}
                            self.rooms_warning.append(checked_room)
                            self.room_name=message

        if self.chosen_owner==0:
            self.chosen_owner=1    
            self.requested_owner=message
            buttons=[]
            for own in self.rooms:
                if own['owner']==self.requested_owner:
                    for room in own['rooms']:
                        buttons.append ([InlineKeyboardButton(text=room["room_name"],callback_data=room["room_name"])])
            keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
            print("Chosen owner: ", self.requested_owner, "\n") #PRINT FOR DEMO
            self.bot.sendMessage(chat_ID, text='Which room are you interested in?', reply_markup=keyboard) 

        elif self.chosen_owner==1 and self.chosen_room==0:
            self.chosen_room=1
            self.requested_room=message
            print("Chosen room: ", self.requested_room, "\n") #PRINT FOR DEMO
            for own in self.rooms:
                if own['owner']==self.requested_owner:
                    for room in own['rooms']:
                        if room['room_name']==self.requested_room:
                            self.r=room
            buttons=[]  
            for own in self.rooms:
                if own['owner']==self.requested_owner:
                    for room in own['rooms']:
                        if room["room_name"]==message or self.room_name==room["room_name"] :
                            for dev in room["room_sensors"]:
                                buttons.append ([InlineKeyboardButton(text=dev,callback_data=dev)])
                                
            keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
                       
        if self.check==0 and self.chosen_owner==1 and self. chosen_room==1:  
            self.bot.sendMessage(chat_ID, text="/operation to choose another room or /check")    
                 

if __name__ == "__main__":
    conf = json.load(open("settings_warning.json"))
    token = conf["telegramToken"]
    service_catalog_info=("service_catalog_info.json")
    bot=EchoBot1(token,service_catalog_info,"health_limits.json")

    print("Bot started ...")
    while True:
        time.sleep(3)