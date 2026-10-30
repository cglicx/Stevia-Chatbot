# Stevia Chatbot

A simple local Flask chatbot that answers common Stevia plant-care questions from a small knowledge base. It uses a locally trained multilingual character- and word-ngram classifier to identify the likely topic of each question. A keyword matcher is retained as a deterministic fallback if the local model cannot be loaded.

## Run locally

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   python app.py
   ```

4. Open http://127.0.0.1:5000 in a browser.

The trained classifier and its TF-IDF vocabularies are included in `models/stevia-intent-bert` as a local joblib model, so startup does not require a model download. The app still answers using the keyword and knowledge-base fallback if the local model is unavailable. A uniquely recognized one-edit typo in a Stevia-care or sensor term is corrected silently. A uniquely recognized typo requiring two edits, or multiple typos in one request, prompts the user to confirm the complete suggested correction; ambiguous or unrelated words are left unchanged. Reply "yes" to use the suggested request, or "no" to leave it unchanged and rephrase.

The chatbot also recognizes common Tagalog and Taglish Stevia-care terms and questions, routes them through the same care and sensor rules, and responds in Tagalog or casual Taglish with a light, friendly tone when Filipino wording is detected. The intent-training CSV includes roughly 40-42 Tagalog and Taglish examples for each main care topic (26 each for sensor records), alongside the English examples, for future model retraining. This is a rule-based vocabulary and response set, not general-purpose translation.

Light and pump requests in English, Tagalog, and Taglish prompt for confirmation before sending a command. The connected ESP32 sketch controls only those two relays; fan requests are explicitly reported as unsupported and are not sent. The Yes button or a typed affirmative reply sends the confirmed command; No sends nothing. Commands go to `POST /api/device-command`, which forwards canonical `device` (`light` or `pump`) and `action` (`ON` or `OFF`) values to the PHP backend. After a command is accepted, the chatbot checks the PHP `device-status` route and reports the latest `actual_state`; it does not claim the device is on if that state has not been reported or still differs from the requested state. "Sprinkler" maps to `pump`. Do not expose this unauthenticated device-control endpoint beyond a trusted network.

## Retraining and evaluation

Retrain the intent model with a grouped, stratified 70/15/15 train/validation/test split. Device-control phrases are represented in Tagalog and Taglish alongside the care intents. Batch size 16 provides frequent updates for the smaller intent classes, with a 50-epoch maximum and validation-loss early stopping:

```powershell
python train_intent_classifier.py --data intent_training_data.csv --output-dir models/stevia-intent-bert --epochs 50 --batch-size 16 --learning-rate 0.05 --alpha 1e-5 --patience 7 --seed 42
```

Training uses a grouped, stratified 70/15/15 train/validation/test split to reduce split leakage. The multilingual character and word n-gram vocabularies are learned from the training set only, after common Tagalog/Taglish terms are normalized. Minibatch SGD uses validation-loss early stopping and saves the best checkpoint; the validation set is then used for confidence calibration and for choosing when a confident model prediction should override keyword routing. The model, vocabularies, configuration, split indices, per-intent classification report, routing metrics, and English/Tagalog/Taglish smoke-test results—including device-control scores—are saved alongside the evaluation report. The recommended batch size is 16, maximum epoch count is 50, learning rate is 0.05, regularization alpha is 1e-5, and patience is 7; these CPU-friendly settings avoid downloading a large pretrained model. Do not treat metrics on synthetic training phrases as production accuracy: evaluate on independently collected, consented user utterances and multi-turn conversations before deployment.

## FastAPI mobile API

The FastAPI API can run alongside the Flask web interface. Install the requirements, then start it with:

```powershell
py -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

If PowerShell still reports that `uvicorn` is not recognized, use the Python launcher explicitly (`py -m uvicorn ...`) or activate the project virtual environment before running the command.

Send a JSON `POST` to `http://<server-address>:8000/api/chat` with `message` and optional `history`. The PHP backend supplies current sensor context from `capstone_db.sensor_readings`; direct FastAPI requests fetch the same context from PHP when it is omitted. Configure `CAPSTONE_BACKEND_API_BASE_URL` to the backend entry point, including its trailing `url=` query parameter, when the backend is not at the default address. For a phone on the same Wi-Fi, use the computer's LAN IP address and allow port `8000` through its firewall. For an Android emulator, the host computer is usually available at `10.0.2.2`. Do not use `localhost` from a physical phone. Deploy behind HTTPS and add authentication before exposing the API publicly.

## Uploaded knowledge

The included `sprout_knowledge_base.json` is loaded at startup as the primary SPROUT knowledge source. It contains the Stevia parameters, valid ranges, classifications, interpretations, recommendations, and sources. Normal answers for soil pH, temperature, humidity, light, and watering are generated from these JSON records. `sprout_knowledge_base.csv` is supported as a fallback if the JSON file is unavailable. The intent classifier identifies the question topic; the chatbot then retrieves the relevant uploaded knowledge and uses it to explain the answer or a sensor reading.

For example, sending `{"temperature": 32}` with a temperature question returns the uploaded `heat_stress` classification and its recommended shading, ventilation, and irrigation action.

## Sensor data

The `sensor_readings/` folder contains Excel files for individual sensor parameters:
- `Environment Temperature.xlsx` - ambient temperature readings
- `Environment Humidity.xlsx` - ambient humidity
- `Soil Moisture.xlsx` - soil water level
- `Soil pH.xlsx` - soil pH readings
- `Soil Temperature.xlsx` - soil temperature
- `Environment Light Intensity.xlsx` - light intensity in lux
- `Water TDS.xlsx` - water quality (total dissolved solids)
- `Solar Panel Battery Voltage.xlsx` - power system status

The mobile API path and the Flask web API fetch the latest reading and up to 1,000 records from the PHP backend; they do not fall back to the Excel/CSV snapshot when the backend is unavailable. A failed backend request is returned as an API error. The Excel files and `IoTProcessed_Data.csv` remain available for local chatbot utility and tests, but are not the source of live mobile readings. For Stevia care questions, deterministic rules compare compatible sensor values with the SPROUT target ranges; they do not use a separate forecast model. Light is recorded in lux while SPROUT's light target is daily sun-hours, so the chatbot reports that mismatch instead of comparing unlike units. Soil moisture is reported as a percentage, but SPROUT provides qualitative wet/dry guidance without calibrated numeric thresholds; the chatbot therefore bases watering advice on checking the top inch rather than inventing a percentage cutoff. A request for "the latest sensor readings" asks how many records to show; replying with a number returns those database records, newest first. Historical record retrieval is limited to the latest 1,000 records in the PHP/FastAPI chat path and explicitly reports when the database contains more. A singular latest-reading request returns the current database snapshot.

`POST /api/chat` accepts `message` and optional `history`; optional `sensor_data` can be used by a trusted integration, but the normal mobile flow supplies the live database snapshot:

```json
{
  "message": "Should I water my Stevia?",
  "sensor_data": {
    "temperature": 24,
    "humidity": 58,
    "soil_moisture": 42,
    "light_intensity": 780
  }
}
```

The browser also sends the last few turns as `history`, allowing short follow-up questions to retain context:

```json
{
   "message": "What about indoors?",
   "history": [
      {"role": "user", "content": "What temperature is best for Stevia?"},
      {"role": "assistant", "content": "Stevia prefers roughly 18-30 C..."}
   ]
}
```

For mobile clients, a device confirmation can also include the `device_command` object returned by the initial `/api/chat` response. Send that object as `pending_device_command` with the user's reply so confirmation works even if the app does not resend conversation history:

```json
{
  "message": "Yes",
  "pending_device_command": {
    "device": "light",
    "action": "ON"
  }
}
```

Only an affirmative reply executes the pending command; `"No"` cancels it. The mobile app should retain the returned `device_command` until the user chooses an option, pass it with the confirmation message, then append the returned answer as the assistant's next chat turn. The web client already follows this flow.

Each knowledge-base topic has multiple answer variants. The chatbot checks recent assistant messages and selects an unused variant, changing the structure between direct guidance, a checklist or range summary, and troubleshooting advice. After all variants for a topic have been used, it cycles back to the first grounded answer.

The live data flow is ESP32 → `POST /Capstone/Backend/index.php?url=sensor` → MySQL `sensor_readings` → PHP sensor read endpoints → mobile app and AI assistant. Set `CAPSTONE_BACKEND_API_BASE_URL` in the chatbot process to the computer's LAN address, for example `http://<computer-lan-ip>/Capstone/Backend/index.php?url=`. The ESP32 sketch in this project is not checked in; update its endpoint to the actual PHP folder path and use the computer's IP address. If the PHP or AI service cannot be reached, clients receive an error instead of answers based on old files.

## Project structure

- `app.py` - Flask routes and application entry point
- `stevia_bot/knowledge_base.py` - editable Stevia facts and suggested questions
- `stevia_bot/chatbot.py` - local intent classification, fallback matching, and sensor boundary
- `stevia_bot/iot_data.py` - sensor_readings Excel loader with CSV fallback and reading normalization
- `sensor_readings/` - individual Excel files for each sensor parameter
- `templates/index.html` - chat interface
- `static/style.css` - interface styling
- `static/app.js` - chat behavior
