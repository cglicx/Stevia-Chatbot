from stevia_bot.chatbot import SteviaChatbot

history = [
    {'role': 'assistant', 'content': 'Latest snapshot (2024-09-27 13:28:26): temperature 26.7 C, humidity 95.0%, and water level 64.05%.'},
    {'role': 'assistant', 'content': 'The newest recorded data is dated 2024-09-27 13:28:26. Temperature is 26.7 C, humidity is 95.0%, and the water level is 64.05%.'},
]
summary = SteviaChatbot._current_reading_summary({
    'recorded_at': '2024-09-27 13:28:26',
    'temperature': 26.7,
    'humidity': 95.0,
    'water_level': 64.05,
}, history)
print('SUMMARY:', summary)
print('WATERING:', SteviaChatbot._choose_response('watering', 'How often should I water it?', history))
print('TEMP:', SteviaChatbot._choose_response('temperature', 'What temperature is best for Stevia?', history))
