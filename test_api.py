#!/usr/bin/env python
from app import app, chatbot

chatbot._model_loaded = True
chatbot.model = None
client = app.test_client()

data = client.post('/api/chat', json={'message': 'What are the latest sensor readings?'}).get_json()
print("API Test Results:")
print(f"  Status: {'OK' if data.get('answer') else 'ERROR'}")
print(f"  Knowledge source: {data.get('knowledge_source')}")
print(f"  Sensor source: {data['sensor_data'].get('source')}")
print(f"  Answer preview: {data.get('answer', '')[:150]}...")
print(f"\nSensor data keys: {list(data['sensor_data'].keys())}")
