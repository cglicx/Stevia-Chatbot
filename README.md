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

Device requests in English, Tagalog, and Taglish (for example, "open fan," "buksan ang bentilador," "i-off mo yung fan," and "patayin ang ilaw") prompt for confirmation before sending a command. These phrases are also included under the `device_control` intent in the training CSV. The Yes button or a typed affirmative reply sends the confirmed command; No sends nothing. Commands go to `POST /api/device-command`, which forwards canonical `device` (`light`, `fan`, or `pump`) and `action` (`ON` or `OFF`) values to the configured backend. Set `DEVICE_CONTROL_API_URL` and `DEVICE_STATUS_API_URL` to backend URLs reachable from this service; there are no private-network defaults. Optionally set `DEVICE_BACKEND_TOKEN` if the backend accepts bearer-token authentication. "Sprinkler" maps to `pump`. Device control is unavailable until a reachable backend is configured.

## Retraining and evaluation

Install the optional training dependencies before retraining:

```powershell
python -m pip install -r requirements-training.txt
```

Retrain the intent model with a grouped, stratified 70/15/15 train/validation/test split. Device-control phrases are represented in Tagalog and Taglish alongside the care intents. Batch size 16 provides frequent updates for the smaller intent classes, with a 50-epoch maximum and validation-loss early stopping:

```powershell
python train_intent_classifier.py --data intent_training_data.csv --output-dir models/stevia-intent-bert --epochs 50 --batch-size 16 --learning-rate 0.05 --alpha 1e-5 --patience 7 --seed 42
```

Training uses a grouped, stratified 70/15/15 train/validation/test split to reduce split leakage. The multilingual character and word n-gram vocabularies are learned from the training set only, after common Tagalog/Taglish terms are normalized. Minibatch SGD uses validation-loss early stopping and saves the best checkpoint; the validation set is then used for confidence calibration and for choosing when a confident model prediction should override keyword routing. The model, vocabularies, configuration, split indices, per-intent classification report, routing metrics, and English/Tagalog/Taglish smoke-test results—including device-control scores—are saved alongside the evaluation report. The recommended batch size is 16, maximum epoch count is 50, learning rate is 0.05, regularization alpha is 1e-5, and patience is 7; these CPU-friendly settings avoid downloading a large pretrained model. Do not treat metrics on synthetic training phrases as production accuracy: evaluate on independently collected, consented user utterances and multi-turn conversations before deployment.

## FastAPI mobile API and Render deployment

The mobile API is a separate service from the Flask browser interface. Run it locally with:

```powershell
py -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

The `render.yaml` Blueprint deploys the FastAPI service using `requirements-render.txt`, binds to Render's `$PORT`, and uses `/healthz` for health checks. In Render, the Blueprint generates a `CHATBOT_API_KEY` and sets `ENVIRONMENT=production`. Send the key with every API request as `Authorization: Bearer <CHATBOT_API_KEY>`; never commit it. Production startup fails if the key is missing or shorter than 32 characters. The health endpoint is public; `/api/*` routes require the key. A shared key packaged in a public mobile app can be extracted, so this is a basic access gate, not per-user authentication. For a public consumer release, integrate user authentication and verify user tokens on the API.

For local development, API authentication is optional unless `CHATBOT_API_KEY` is set. To configure browser-based clients, set `CORS_ORIGINS` to a comma-separated list of exact origins; native mobile apps do not normally need CORS.

Send a JSON `POST` to `https://<your-render-service>.onrender.com/api/chat` with the same `message`, optional `sensor_data`, and optional `history` fields described below. Include the bearer token header on `/api/chat` and `/api/device-command`. A device-control request is forwarded only after confirmation, but its backend must be hosted somewhere Render can reach; a `192.168.x.x` address on your local network will not work. Set `DEVICE_CONTROL_API_URL` and `DEVICE_STATUS_API_URL` in the Render dashboard, and optionally `DEVICE_BACKEND_TOKEN`. If these URLs are not configured, device-control requests return HTTP 503 rather than silently pretending to work. Set `OPENAI_API_KEY` in Render only if you want to enable OpenAI-backed responses; the local classifier and deterministic fallback work without it.

Browser clients can omit the bearer key when their request is same-origin or comes from an exact origin listed in `CORS_ORIGINS`; configure the frontend's origin there. This is only a browser integration convenience: an origin header is not authentication and can be forged by non-browser clients. Never put the shared key in public frontend code. Add user authentication and rate limiting before exposing the API as a public consumer service.

## Vercel API deployment

The root `pyproject.toml` points Vercel at `fastapi_app:app`, avoiding the separate Flask entrypoint in `app.py`. Deploy this repository as a Vercel project; the Python runtime installs the lean runtime packages from `requirements.txt`. The included joblib intent classifier is used in production. Install `requirements-training.txt` locally only when running classifier training or the optional Transformers fallback.

In the Vercel project settings, add these environment variables for Production (and Preview if needed):

- `ENVIRONMENT` = `production`
- `CHATBOT_API_KEY` = a randomly generated secret with at least 32 characters
- `CORS_ORIGINS` = comma-separated exact origins of browser frontends that call this API, such as `https://your-app.example`

The API is available at `https://<your-vercel-domain>/api/chat`; `/healthz` is public. Browser requests from an origin in `CORS_ORIGINS` (or from the API's own origin) can call it without exposing `CHATBOT_API_KEY` in frontend code. Keep the key for server-to-server callers. This origin check is not an access-control boundary; add authenticated user tokens and rate limits for a public application.

If you use the included Flask browser frontend on a separate deployment, set `CHAT_API_BASE_URL` on that frontend's server to the Vercel API origin (without a trailing slash). Add the frontend's exact origin to `CORS_ORIGINS` on Vercel. When the frontend and API are on the same origin, leave `CHAT_API_BASE_URL` unset. Set `DEVICE_CONTROL_API_URL` (and optionally `DEVICE_STATUS_API_URL` and `DEVICE_BACKEND_TOKEN`) in Vercel only when the device backend is reachable from Vercel; local-network device URLs will not work.

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

For mobile clients, a device confirmation response includes button choices in `confirmation.options`, `quick_replies`, and `buttons`, plus the canonical device command as `device_command` (also `deviceCommand`). The mobile UI must render those button choices. Send the command object as `pending_device_command` or `pendingDeviceCommand` with the user's reply so confirmation works even if the app does not resend conversation history:

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

Sensor integration is intentionally not required. The chatbot service already receives optional readings and adds relevant context for watering, temperature, humidity, and light questions. A future IoT adapter can call this API or populate the same dictionary from MQTT, serial, or an HTTP device service.

## Project structure

- `app.py` - Flask routes and application entry point
- `stevia_bot/knowledge_base.py` - editable Stevia facts and suggested questions
- `stevia_bot/chatbot.py` - local intent classification, fallback matching, and sensor boundary
- `stevia_bot/iot_data.py` - sensor_readings Excel loader with CSV fallback and reading normalization
- `sensor_readings/` - individual Excel files for each sensor parameter
- `templates/index.html` - chat interface
- `static/style.css` - interface styling
- `static/app.js` - chat behavior
