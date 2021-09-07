import constants as keys;
from telegram.ext import *
import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "levi"
print("Let's chat! (type 'quit' to exit)")

def start_command(update, context):
    update.message.reply_text("hi there! type anything you like")


def help_command(update, context):
    update.message.reply_text("try typing something else maybe?")

def error(update, context):
    print(f"Update {update} caused error {context.error}")

def handleInput(sentence):   
    if sentence == "quit":
        print(f"{bot_name}: see you!")

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                return str(random.choice(intent['responses']))
    else:
        print(f"{bot_name}: i'm sorry could you type something else")
        return "i'm sorry could you type something else"

def handle_message(update, context):
    sentence = str(update.message.text).lower();
    response = handleInput(sentence)

    update.message.reply_text(response)

def main():
    updater = Updater(keys.API_KEY, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    
    #dp.add_error_handler(error)

    updater.start_polling() # code that starts the bot
    updater.idle() # to ensure that bot continues to stay active

main()
