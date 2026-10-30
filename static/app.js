const form = document.querySelector('#chat-form');
const input = document.querySelector('#message');
const messages = document.querySelector('#messages');
const suggestions = document.querySelectorAll('.suggestion');
const conversation = [];

const sensorRecordColumns = [
  ['recorded_at', 'Recorded at'],
  ['temperature', 'Temperature (C)'],
  ['humidity', 'Humidity (%)'],
  ['water_level', 'Soil moisture (%)'],
  ['soil_ph', 'Soil pH'],
  ['soil_temperature', 'Soil temperature (C)'],
  ['light_intensity', 'Light intensity (lux)'],
  ['water_tds', 'Water TDS (ppm)'],
  ['battery_voltage', 'Battery voltage (V)'],
  ['nitrogen', 'Nitrogen'],
  ['phosphorus', 'Phosphorus'],
  ['potassium', 'Potassium'],
  ['fan_on', 'Fan'],
  ['watering_pump_on', 'Watering pump'],
  ['water_pump_on', 'Water pump'],
];

function addMessage(text, type, meta = '') {
  const article = document.createElement('article');
  article.className = `message ${type}`;
  if (type === 'bot') {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'S';
    article.appendChild(avatar);
  }

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  bubble.appendChild(paragraph);

  if (meta) {
    const metadata = document.createElement('small');
    metadata.textContent = meta;
    bubble.appendChild(metadata);
  }

  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function addSensorRecordsTable(container, records) {
  if (!Array.isArray(records) || records.length === 0) return;

  const availableColumns = sensorRecordColumns.filter(([key]) =>
    records.some((record) => Object.hasOwn(record, key))
  );
  const wrapper = document.createElement('div');
  wrapper.className = 'sensor-records-table-wrap';
  const table = document.createElement('table');
  table.className = 'sensor-records-table';
  const caption = document.createElement('caption');
  caption.textContent = `${records.length} sensor ${records.length === 1 ? 'record' : 'records'}`;
  table.appendChild(caption);

  const head = document.createElement('thead');
  const headerRow = document.createElement('tr');
  availableColumns.forEach(([, label]) => {
    const header = document.createElement('th');
    header.scope = 'col';
    header.textContent = label;
    headerRow.appendChild(header);
  });
  head.appendChild(headerRow);
  table.appendChild(head);

  const body = document.createElement('tbody');
  records.forEach((record) => {
    const row = document.createElement('tr');
    availableColumns.forEach(([key]) => {
      const cell = document.createElement('td');
      const value = record[key];
      cell.textContent = key.endsWith('_on') && typeof value === 'boolean'
        ? (value ? 'On' : 'Off')
        : (value === null || value === undefined ? '—' : String(value));
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  table.appendChild(body);
  wrapper.appendChild(table);
  container.appendChild(wrapper);
}

function addDeviceConfirmation(container, data) {
  const command = data.device_command || data.deviceCommand;
  const options = data.confirmation?.options || data.quick_replies || data.buttons;
  if (!data.confirmation?.required || !command || !Array.isArray(options)) return;

  const controls = document.createElement('div');
  controls.className = 'device-confirmation';
  options.forEach((option) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = option.label || option.text;
    button.addEventListener('click', () => {
      handleDeviceConfirmation(controls, command, option.value);
    });
    controls.appendChild(button);
  });
  container.appendChild(controls);
}

function handleDeviceConfirmation(controls, command, choice) {
  controls.querySelectorAll('button').forEach((item) => {
    item.disabled = true;
  });
  ask(choice === 'yes' ? 'Yes' : 'No', command);
}

async function ask(question, pendingDeviceCommand = null) {
  addMessage(question, 'user');
  conversation.push({ role: 'user', content: question });
  const pending = addMessage('Thinking through the plant notes...', 'bot', 'BERT · matching topic');
  input.value = '';
  input.disabled = true;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: question,
        history: conversation.slice(-8),
        ...(pendingDeviceCommand ? { pending_device_command: pendingDeviceCommand } : {}),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to answer right now.');
    pending.querySelector('p').textContent = data.answer;
    addSensorRecordsTable(pending.querySelector('.bubble'), data.sensor_records);
    addDeviceConfirmation(pending.querySelector('.bubble'), data);
    pending.querySelector('small').textContent = `${data.topic} · ${data.source}`;
    conversation.push({ role: 'assistant', content: data.answer });
  } catch (error) {
    pending.querySelector('p').textContent = error instanceof Error
      ? error.message
      : 'Connection error. Please try again.';
    pending.querySelector('small').textContent = 'Connection issue';
  } finally {
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (input.value.trim()) ask(input.value.trim());
});
suggestions.forEach((button) => button.addEventListener('click', () => ask(button.textContent)));
