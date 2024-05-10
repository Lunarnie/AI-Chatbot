import random
import json  # dùng để đọc Key-Value
import pickle #(chuyển đổi) các đối tượng Python thành dạng dữ liệu nhị phân và ngược lại
import numpy as np
import tensorflow as tf
import nltk
#Natural Language Toolkit là một trong những thư viện open-source xử lí ngôn ngữ tự nhiên.
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() #dùng để thực hiện từ nguyên hóa -> quá trình chuyển các từ về dạng cơ bản hoặc dạng gốc
                                 # VD: Running -> Run / am/is/are -> be

#from neuralintents.assistants import BasicAssistant
#intents = BasicAssistant(A)
intents = json.loads(open('intents.json').read())

words = []  #chứa từ trong patterns
classes = [] #chứa tag trong patterns
documents = [] #kết hợp chứa cả (từ,tag)
ignoreLetters = ['?', '!', '.', ','] #bỏ qua các kí tự

#Lặp từng intent trong intents là lặp từng cụm: tag, patterns, reponses
for intent in intents['intents']:
    for pattern in intent['patterns']:  #lặp từng kết quả trong patterns đã tạo 
        wordList = nltk.word_tokenize(pattern)
        #tách một câu hoặc đoạn văn thành các từ, dấu câu và các yếu tố khác
        words.extend(wordList)  #Thêm tất cả giá trị vào cái list mới tạo wordList vd: "Hi,"How are you" -> ['Hi', 'How', 'are', 'you','?'] 
        #Dùng append Chỉ thêm được 1 giá trị vào list sẵn vd: "Hi,"How are you" -> ['Hi', ['How', 'are', 'you','?']]
        documents.append((wordList, intent['tag'])) 
        #Thêm wordlist đã được token hóa vào documents trống, với tag kế bên
        #VD: (['Hi'],'greeting'),(['what','should','I','call','you','name'],'name')...
        #Gắn nhãn -tag vào classes đã tạo trống
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes)) 
'''
Có lệnh set( ) loại các phần tử giống nhau => Dùng cho list, string, tuple được
lệnh sorted -> sắp xếp các phần tử ở trong một list, set hay tuple.
Các ký tự in hoa sẽ có giá trị và thứ tự khác với các ký tự không in hoa.
In hoa ưu tiên ra đầu. Sắp xếp các từ theo kí tự đầu - alphabet

'''

#lưu trữ data vào file .pkl -> khôi phục danh sách từ file dùng pickle.load()
#wb là mở file trong chế độ ghi nhị phân thay vì văn bản
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes) #len-đếm số phần tử trong các đối tượng khác nhau

for document in documents:
    bag = []
    wordPatterns = document[0] 
    '''
    vd: (['Hi'],'greeting'),(['what','should','I','call','you','name'],'name')...
    ->document[0]=['Hi'],['what','should','I','call','you','name'] 
    document[1]='greeting','name'

    '''
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1 #xem thuộc classes nào vd thuộc greeting 
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation='softmax'))

from keras.optimizers import SGD
opt = SGD(learning_rate=0.01,momentum=0.9)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainX, trainY, epochs=200, batch_size=10, verbose=1)
model.save('chatbot_model.h5')
print('Done')

