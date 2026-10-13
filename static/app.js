const form = document.querySelector('#chat-form');
const input = document.querySelector('#message');
const messages = document.querySelector('#messages');
const suggestions = document.querySelectorAll('.suggestion');
const conversation = [];

function addMessage(text, type, meta = '') {
  const article = document.createElement('article');
  article.className = `message ${type}`;
  article.innerHTML = type === 'bot'
    ? `<div class="avatar">S</div><div class="bubble"><p></p>${meta ? `<small>${meta}</small>` : ''}</div>`
    : `<div class="bubble"><p></p></div>`;
  article.querySelector('p').textContent = text;
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

async function ask(question) {
  addMessage(question, 'user');
  conversation.push({role: 'user', content: question});
  const pending = addMessage('Thinking through the plant notes...', 'bot', 'BERT · matching topic');
  input.value = '';
  input.disabled = true;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: question, history: conversation.slice(-8)})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to answer right now.');
    pending.querySelector('p').textContent = data.answer;
    pending.querySelector('small').textContent = `${data.topic} · ${data.source}`;
    conversation.push({role: 'assistant', content: data.answer});
  } catch (error) {
    pending.querySelector('p').textContent = error.message;
    pending.querySelector('small').textContent = 'Connection issue';
  } finally {
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', (event) => { event.preventDefault(); if (input.value.trim()) ask(input.value.trim()); });
suggestions.forEach((button) => button.addEventListener('click', () => ask(button.textContent)));
