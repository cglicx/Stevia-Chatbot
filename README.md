# Stevia BERT Chatbot

A simple local Flask chatbot that answers common Stevia plant-care questions from a small knowledge base. It uses a fine-tuned BERT sequence classifier to identify the likely topic of each question. A keyword matcher is retained as a deterministic fallback if the local model cannot be loaded.

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

The trained classifier and tokenizer are included in `models/stevia-intent-bert`, so startup does not require a model download. The app still answers using the keyword and knowledge-base fallback if the local model is unavailable. A uniquely recognized one-edit typo in a Stevia-care or sensor term is corrected silently. A uniquely recognized typo requiring two edits, or multiple typos in one request, prompts the user to confirm the complete suggested correction; ambiguous or unrelated words are left unchanged. Reply "yes" to use the suggested request, or "no" to leave it unchanged and rephrase.

## Retraining and evaluation

Retrain the intent model with:

```powershell
python train_intent_classifier.py --data intent_training_data.csv --output-dir models/stevia-intent-bert --epochs 8 --batch-size 8 --learning-rate 2e-5 --patience 2 --seed 42
```

Training fine-tunes the complete BERT model, uses a stratified 70/15/15 train/validation/test split, groups highly similar same-intent phrases to reduce split leakage, and stops when validation loss does not improve. The validation set is used for confidence calibration and for choosing when a confident model prediction should override keyword routing. The report includes both classifier-only metrics and end-to-end intent-routing metrics. Do not treat metrics on synthetic training phrases as production accuracy: evaluate on independently collected, consented user utterances and multi-turn conversations before deployment.

## FastAPI mobile API

The FastAPI API can run alongside the Flask web interface. Install the requirements, then start it with:

```powershell
py -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

If PowerShell still reports that `uvicorn` is not recognized, use the Python launcher explicitly (`py -m uvicorn ...`) or activate the project virtual environment before running the command.

Send a JSON `POST` to `http://<server-address>:8000/api/chat` with the same `message`, optional `sensor_data`, and optional `history` fields described below. For a phone on the same Wi-Fi, use the computer's LAN IP address and allow port `8000` through its firewall. For an Android emulator, the host computer is usually available at `10.0.2.2`. Do not use `localhost` from a physical phone. Deploy behind HTTPS and add authentication before exposing the API publicly.

## Uploaded knowledge

The included `sprout_knowledge_base.json` is loaded at startup as the primary SPROUT knowledge source. It contains the Stevia parameters, valid ranges, classifications, interpretations, recommendations, and sources. Normal answers for soil pH, temperature, humidity, light, and watering are generated from these JSON records. `sprout_knowledge_base.csv` is supported as a fallback if the JSON file is unavailable. BERT identifies the question topic; the chatbot then retrieves the relevant uploaded knowledge and uses it to explain the answer or a sensor reading.

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

The chatbot loads the latest values and measurement units from these Excel files automatically at startup. If the folder is not available, it falls back to `IoTProcessed_Data.csv`. For Stevia care questions, deterministic rules compare compatible sensor values with the SPROUT target ranges and recommendations; they do not use an additional recommendation model. Light is recorded in lux while SPROUT's light target is daily sun-hours, so the chatbot reports that mismatch instead of comparing unlike units. Soil moisture is reported as a percentage, but SPROUT provides qualitative wet/dry guidance without calibrated numeric thresholds; the chatbot therefore bases watering advice on checking the top inch rather than inventing a percentage cutoff. A request for "the latest sensor readings" asks how many records to show; replying with a number (including "at least 3") returns that many historical records, newest first, in a table. Requests with a number included return the selected records immediately. Short follow-ups such as "the 3 latest," "show me the next three," and "put them in a table" use the recent sensor-record context, while a clear new question can change topics. A singular latest-reading request returns the current snapshot. Manual `sensor_data` values sent to the API override matching dataset values, which keeps live IoT integration straightforward.

`POST /api/chat` accepts an optional `sensor_data` object in addition to `message`:

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

Each knowledge-base topic has multiple answer variants. The chatbot checks recent assistant messages and selects an unused variant, changing the structure between direct guidance, a checklist or range summary, and troubleshooting advice. After all variants for a topic have been used, it cycles back to the first grounded answer.

Sensor integration is intentionally not required. The chatbot service already receives optional readings and adds relevant context for watering, temperature, humidity, and light questions. A future IoT adapter can call this API or populate the same dictionary from MQTT, serial, or an HTTP device service.

## Project structure

- `app.py` - Flask routes and application entry point
- `stevia_bot/knowledge_base.py` - editable Stevia facts and suggested questions
- `stevia_bot/chatbot.py` - fine-tuned BERT intent classification, fallback matching, and sensor boundary
- `stevia_bot/iot_data.py` - sensor_readings Excel loader with CSV fallback and reading normalization
- `sensor_readings/` - individual Excel files for each sensor parameter
- `templates/index.html` - chat interface
- `static/style.css` - interface styling
- `static/app.js` - chat behavior
