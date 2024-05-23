import random
import json  # dùng để đọc Key-Value
import pickle #(chuyển đổi) các đối tượng Python thành dạng dữ liệu nhị phân và ngược lại
import numpy as np
from keras.models import load_model
import nltk
from langdetect import detect
#Natural Language Toolkit là một trong những thư viện open-source xử lí ngôn ngữ tự nhiên.
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from gtts import gTTS
import pyttsx3
from youtube_search import YoutubeSearch
import webbrowser
import requests
import json
import datetime
import wikipedia
from googletrans import Translator

translator = Translator()
lemmatizer = WordNetLemmatizer() #dùng để thực hiện từ nguyên hóa -> quá trình chuyển các từ về dạng cơ bản hoặc dạng gốc
                                 # VD: Running -> Run / am/is/are -> be

#from neuralintents.assistants import BasicAssistant
#intents = BasicAssistant(A)

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('C:/Users/Admin/Downloads/AI-Chatbot-main/chatbot_model.h5')



def is_connected():
    try:
        # Check if we can reach Google's website
        requests.get('https://www.google.com', timeout=5)
        return True
    except requests.ConnectionError:
        return False

def clean_up_sentence(sentence):
    """
  Hàm này thực hiện việc tiền xử lý câu, bao gồm:
    - Tách câu thành các từ riêng lẻ (tokenization)
    - Chuyển các từ về dạng chuẩn (lemmatization)
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """
  Hàm này tạo ra một "túi từ" (bag of words) biểu diễn sự xuất hiện của các từ trong câu.
    """
    sentence_words = clean_up_sentence(sentence)
    bag =[0] * len(words) #Khởi tạo một mảng với kích thước bằng số lượng từ trong vốn từ (words)
    for w in sentence_words:
        '''
        Enumerate là một built-in function của Python. Bạn có thể sử dụng nó cho các vòng lặp của một iterable 
        với việc tự động sinh ra chỉ số index.
        Số đếm (index) bắt đầu mặc định từ 0, nhưng bạn có thể đặt lại nó bằng bất kỳ một số integer nào
        '''
        for i, word in enumerate(words): # Tìm kiếm vị trí của từ hiện tại (w) trong vốn từ (words) - i là vị trí tương ứng của từ trong words
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """
  Hàm này dự đoán lớp (ý định) của một câu đầu vào.
    """
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] #giá trị là phần tử đầu tiên của kết quả dự đoán, 
                                            #tức là xác suất dự đoán của lớp có khả năng cao nhất cho câu đầu vào.
    ERROR_THRESHOLD = 0.25 #ngưỡng giá trị xác suất của từng lớp - tự cho là 0,25 nếu xác suất lớn hơn số này thì mới nhận, còn xác suất lớp nhỏ hơn loại
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] #chỉ lưu trữ những xác suất lớn hơn ngưỡng
    '''
Mỗi phần tử trong results là một danh sách con chứa chỉ số lớp i 
và xác suất dự đoán tương ứng r

    '''
    results.sort(key=lambda x: x[1], reverse=True)# Sắp xếp các lớp theo xác suất dự đoán giảm dần
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability':str(r[1])})
    return return_list # đại diện cho các lớp tiềm năng và xác suất liên quan của chúng cho câu đầu vào.

def lay_chu_sau(van_ban, tu_chi_dinh):
    # Tách văn bản thành danh sách các từ
    cac_tu = van_ban.split()
    
    # Tìm vị trí của từ chỉ định
    vi_tri = cac_tu.index(tu_chi_dinh)
    
    # Nếu từ chỉ định không tìm thấy, trả về chuỗi rỗng
    if vi_tri == -1:
        return ""
    
    # Lấy chữ sau từ chỉ định 

    if vi_tri < len(cac_tu) - 1:
        phrase_or_sentence = " ".join(cac_tu[vi_tri + 1:])
        return phrase_or_sentence
    else:
        return ""

def get_response(intents_list, intents_json):
    """
  Hàm này lấy câu trả lời thích hợp dựa trên lớp (ý định) được dự đoán.
    """
    tag = intents_list[0]['intent'] #biến tag được gán vào phần tử đầu tiên trong intents_list, 
    # kết quả là classes[r[0]] - lớp có xác suất cao nhất tương ứng với câu đầu vào
    #Trích xuất nhãn được dự đoán cao nhất cho câu đầu vào từ danh sách intents_list và lưu trữ nó trong biến tag
    list_of_intents = intents_json['intents']
    for i in list_of_intents: #cho chạy lặp trong list intents_json, lấy kết quả
        if i['tag'] == tag: #Nếu biến kết quả của tag trong list_of_intents trùng với tag trên
            result = random.choice(i['responses'])
            break
    return result

def search_gg():
    try:
        from googlesearch import search
    except ImportError: 
        print("No module named 'google' found")
    for j in search(query, tld="co.in", num=1, stop=1, pause=2):
        print(webbrowser.open(j))

def get_time(message):
    now = datetime.datetime.now()    
    if "time" in message:
        A = bot.say("Right now, the time is %d  %d on %d  %d  %d  in Ho Chi Minh City." %  (now.hour, now.minute,now.day, now.month, now.year))
        translation = translator.translate(A,dest='vi')
        print(translation.text)
 

def weather_forecast(city_name) :       
        API_Key = "fe8d8c65cf345889139d8e545f57819a"
        url = f"http://api.openweathermap.org/data/2.5/weather?"
        call_url = url + "appid=" + API_Key + "&q=" + city_name + "&units=metric"
        response = requests.get(call_url)
        res = response.json()
        if res["cod"] != "404":
               # Storing the live temperature data
            temperature = res["main"]["temp"]
        # Storing the live pressure data
            pressure = res["main"]["pressure"]
            desc = res["weather"]
            # Storing the weather description
            weather_description = desc[0]["description"]
            bot.say("Temperature (in celsius scale): " + str(temperature))
            bot.say("Pressure: " + str(pressure))
            bot.say("Description: " + str(weather_description))
            print("Temperature (in celsius scale): " + str(temperature))
            print("Pressure: " + str(pressure))
            print("Description: " + str(weather_description))
            if temperature < 20:
               responses =["It's a good idea to bring along a jacket or coat when heading outdoors to stay comfortable and protected from the cool breeze",
               "You need to consider wearing boots or closed-toe shoes to keep your feet warm and protected from the chill",
               "Even on cooler days, UV rays can still be harmful. Remember to apply sunscreen if spending prolonged periods outdoors to protect your skin"]
               Response = random.choice(responses)
               bot.say(Response)
            elif  20 < temperature < 30:
                responses = ["UV rays can be strong, even on moderately warm days. Remember to apply sunscreen with a high SPF to protect your skin from sunburn and UV damage.",
                "Cool showers or baths can help lower your body temperature and provide relief from the heat. Consider taking a refreshing shower to cool off during the day",
                "Light-colored clothing reflects sunlight and heat, helping to keep you cooler compared to dark-colored clothing"]
                Response = random.choice(responses)
                bot.say(Response)
            elif temperature > 30:
                responses = ["Minimize outdoor activities, especially during the hottest part of the day.",
  "Apply sunscreen with a high SPF to exposed skin, and reapply regularly, especially if sweating or swimming. Wear sunglasses to protect your eyes from harmful UV rays",
  "Ensure your pets have access to shade, water, and cool shelter during hot weather. Never leave them in parked cars, as temperatures can quickly become dangerously high.",
  "Choose lightweight, loose-fitting clothing made of breathable fabrics like cotton to help keep your body cool. Light colors can also help reflect sunlight and heat.",
  "If possible, stay indoors in air-conditioned spaces during the hottest part of the day. If you don't have air conditioning at home, consider visiting public buildings like malls or libraries that are air-conditioned."]
                Response = random.choice(responses)
                print(Response)
                bot.say(Response)
                
        else:
            bot.say("Please enter a valid city name")

bot = pyttsx3.init()
rate = bot.getProperty('rate')
bot.setProperty('rate', 130)
volume = bot.getProperty('volume')
bot.setProperty('volume', 1.0)
voices = bot.getProperty('voices')
bot.setProperty('voice', voices[1].id)
bot.say("Hello lady and Gentleman, I'm Cecilia. Service you is my pleasure")
print("Kính chào quý khách, tôi là Cecilia. Rất hân hạnh được phục vụ bạn") #cript giới thiệu
bot.runAndWait()


while True:
    # Check internet connection before proceeding
    if not is_connected():
        print("No internet connection. Please check your network and try again.")
        break  # Exit the loop if there is no internet connection

    # Initialize the recognizer
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # Read the audio data from the default microphone
        print("Recognizing...")
        r.pause_threshold = 1
        r.energy_threshold = 6000
        r.dynamic_energy_threshold = True
        try:
            # Convert speech to text
            audio = r.listen(source, timeout=1, phrase_time_limit=6)
            print("...")
            message = r.recognize_google(audio, language='en')
            print(f"Recognized (English): {message}")
            
            
            
            
            # Predict class and get resp-onse
            ints = predict_class(message)
            res = get_response(ints, intents)
            print(f"Response: {res}")
            bot.say(res)
            
                    # Translate response to Vietnamese
            translation = translator.translate(res, dest='vi')
            print(f"Translated to Vietnamese: {translation.text}")

            # Ghi đè lên file "translation.txt"
            with open('translation.txt', 'w', encoding='utf-8') as f:
                f.write(translation.text)

            
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    
    if "search" in message.lower() :
        query=lay_chu_sau(str(message.lower()),"search")
        search_gg()
    elif "look for" in message.lower():
        query=lay_chu_sau(str(message.lower()),'for')
        # to search
        search_gg()

    elif "play" in message.lower() or "listen"  in message.lower():
        song = lay_chu_sau(str(message.lower()),'play')
        print(song)
        result = YoutubeSearch(song,max_results = 10).to_dict() #10 kết quả đầu tiên
        url = 'https://www.youtube.com' + result[0]['url_suffix']
        webbrowser.open(url)
    elif "turn on" in message.lower():
        song = lay_chu_sau(str(message.lower()),'on')
        print(song)
        result = YoutubeSearch(song,max_results = 10).to_dict() #10 kết quả đầu tiên
        url = 'https://www.youtube.com' + result[0]['url_suffix']
        webbrowser.open(url)
    
    #try:
        #bot.say("Do you want to hear something?")
    elif "information about" in message.lower():
        text = lay_chu_sau(str(message.lower()),'about')
        contents = wikipedia.summary(text).split('\n')
        bot.say(contents[0])
        translation = translator.translate(contents[0],dest='vi')
        print(translation.text)
        bot.say("Thank you for listening!")   
        print("Cảm ơn bạn đã lắng nghe")  

    elif "weather in" in message.lower():
        city_name = lay_chu_sau(str(message.lower()),"in")
        print(city_name)
        weather_forecast(city_name)
    
    get_time(message)    

    bot.runAndWait()
    

