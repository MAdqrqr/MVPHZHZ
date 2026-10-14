const $ = id => document.getElementById(id);
let busy = false;
let uploadBusy = false;
let conversationVersion = 0;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let result;
  try { result = await response.json(); }
  catch { throw new Error('Сервер жауабы оқылмады. Қосылымды тексеріңіз.'); }
  if (!response.ok || !result.ok) throw new Error(result.error?.message || 'Сұрау орындалмады.');
  return result.data;
}

function status(text, error = false) {
  $('uploadStatus').textContent = text;
  $('uploadStatus').classList.toggle('error', error);
}

async function health() {
  try {
    const data = await api('/api/health');
    $('serverStatus').textContent = 'Сервер қосылған';
    $('uploadLimit').textContent = `${data.max_upload_mb} МБ-қа дейін`;
    const notices = [];
    if (!data.llm_configured) notices.push('Жауап генерациясы үшін серверге LLM API кілтін баптау қажет.');
    if (data.embedding_error) notices.push(data.embedding_error);
    if (!data.index_compatible) notices.push('Модель параметрлері өзгерді. Индексті қайта құрыңыз.');
    $('configNotice').textContent = notices.join(' ');
    $('configNotice').hidden = !notices.length;
  } catch {
    $('serverStatus').textContent = 'Қосылым жоқ';
    $('configNotice').hidden = false;
    $('configNotice').textContent = 'Серверге қосылу мүмкін болмады. Бетті жаңартып көріңіз.';
  }
}

async function documents() {
  const list = await api('/api/documents');
  $('documentList').replaceChildren();
  $('documentCount').textContent = list.length;
  if (!list.length) {
    const empty = document.createElement('li');
    empty.className = 'empty-docs';
    empty.textContent = 'Әзірге құжат жоқ';
    $('documentList').append(empty);
  }
  for (const doc of list) {
    const row = document.createElement('li');
    const icon = document.createElement('span'); icon.className = 'document-icon'; icon.textContent = 'PDF';
    const info = document.createElement('div'); info.className = 'document-info';
    const name = document.createElement('strong'); name.textContent = doc.name;
    const detail = document.createElement('small'); detail.textContent = `✓ Индекстелді · ${doc.pages} мәтінді бет · ${doc.chunks} бөлік`;
    info.append(name, detail);
    const remove = document.createElement('button'); remove.className = 'delete-doc'; remove.textContent = '×';
    remove.setAttribute('aria-label', `${doc.name} құжатын жою`);
    remove.addEventListener('click', async () => {
      if (!window.confirm(`«${doc.name}» құжатын және оның индексін жоясыз ба?`)) return;
      remove.disabled = true;
      try { await api(`/api/documents/${doc.id}`, {method: 'DELETE'}); await documents(); status('Құжат жойылды.'); }
      catch (error) { status(error.message, true); remove.disabled = false; }
    });
    row.append(icon, info, remove); $('documentList').append(row);
  }
}

$('pdfFile').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file || uploadBusy) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) { status('PDF құжатын таңдаңыз.', true); return; }
  uploadBusy = true; $('pdfFile').disabled = true; $('reindex').disabled = true;
  status('Құжат оқылуда және индекстелуде… Алғашқы модель жүктеуі уақыт алуы мүмкін.');
  try {
    const data = new FormData(); data.append('file', file);
    const result = await api('/api/documents/upload', {method: 'POST', body: data});
    status(result.duplicate ? 'Бұл құжат бұрын жүктелген.' : `Дайын: ${result.chunks} мәтін бөлігі индекстелді.`);
    await documents();
  } catch (error) { status(error.message, true); }
  finally { uploadBusy = false; $('pdfFile').disabled = false; $('reindex').disabled = false; event.target.value = ''; await health(); }
});

$('reindex').addEventListener('click', async () => {
  if (uploadBusy) return;
  uploadBusy = true; $('reindex').disabled = true; $('pdfFile').disabled = true;
  status('Индекс қайта құрылуда…');
  try { const result = await api('/api/reindex', {method: 'POST'}); status(`${result.documents} құжат, ${result.chunks} бөлік қайта индекстелді.`); await documents(); }
  catch (error) { status(error.message, true); }
  finally { uploadBusy = false; $('reindex').disabled = false; $('pdfFile').disabled = false; await health(); }
});

function message(role, text, error = false) {
  $('welcome').hidden = true;
  const row = document.createElement('article'); row.className = `message ${role}${error ? ' error' : ''}`;
  const avatar = document.createElement('span'); avatar.className = 'avatar'; avatar.textContent = role === 'user' ? 'Сіз' : '✦';
  const bubble = document.createElement('div'); bubble.className = 'bubble';
  const label = document.createElement('strong'); label.className = 'message-label'; label.textContent = role === 'user' ? 'Сіз' : 'Білім AI';
  const content = document.createElement('p'); content.className = 'message-text'; content.textContent = text;
  bubble.append(label, content); row.append(avatar, bubble); $('messages').append(row);
  scrollChat(); return {row, bubble, content};
}

function scrollChat() { const area = document.querySelector('.conversation'); area.scrollTop = area.scrollHeight; }

$('chatForm').addEventListener('submit', async event => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question || busy) return;
  busy = true; const version = conversationVersion;
  $('send').disabled = true; $('question').value = ''; $('question').style.height = '';
  message('user', question);
  const waiting = message('assistant', 'Оқу материалдарын қарап жатырмын…'); waiting.content.classList.add('loading');
  try {
    const result = await api('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question})});
    if (version !== conversationVersion) return;
    waiting.content.classList.remove('loading'); waiting.content.textContent = result.answer;
    if (result.sources.length) {
      const sources = document.createElement('div'); sources.className = 'sources';
      for (const source of result.sources) {
        const details = document.createElement('details');
        const summary = document.createElement('summary'); summary.textContent = `[${source.citation}] ${source.name} · ${source.page}-бет`;
        const excerpt = document.createElement('p'); excerpt.textContent = source.text;
        details.append(summary, excerpt); sources.append(details);
      }
      waiting.bubble.append(sources);
    }
    const timing = document.createElement('small'); timing.className = 'timing'; timing.textContent = `${result.elapsed_seconds} с · ${result.grounded ? 'Дереккөзге сүйенген жауап' : 'Жеткілікті контекст жоқ'}`; waiting.bubble.append(timing);
  } catch (error) {
    if (version !== conversationVersion) return;
    waiting.content.classList.remove('loading'); waiting.content.textContent = error.message; waiting.row.classList.add('error');
  } finally { busy = false; $('send').disabled = false; scrollChat(); $('question').focus(); }
});

$('question').addEventListener('input', event => { event.target.style.height = ''; event.target.style.height = Math.min(event.target.scrollHeight, 160) + 'px'; });
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('chatForm').requestSubmit(); } });
$('newChat').addEventListener('click', () => { conversationVersion++; $('messages').replaceChildren(); $('welcome').hidden = false; $('question').value = ''; $('question').focus(); });
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => { $('question').value = button.dataset.question; $('question').focus(); }));
health(); documents().catch(error => status(error.message, true));
